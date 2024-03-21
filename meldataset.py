import math
import os
import json
import random
import torch
from torchvision.transforms.functional import resize
import torch.utils.data
import numpy as np
import librosa
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram

MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    # if torch.min(y) < -1.:
    #     print('min value is ', torch.min(y))
    # if torch.max(y) > 1.:
    #     print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    # complex tensor as default, then use view_as_real for future pytorch compatibility
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def get_dataset_filelist(a):
    training_files =[]
    validation_files =[]
    total_files = 0

    audio_dir = "dataset/audio"

    with open("filelists/train.txt") as f:
        training_files = f.readlines()
    for i, line in enumerate(training_files):
        spk, basename = line.strip().split('|')
        training_files[i] = f"{audio_dir}/{spk}/{basename}.wav"

    with open("filelists/val.txt") as f:
        validation_files = f.readlines()
    for i, line in enumerate(validation_files):
        spk, basename = line.strip().split('|')
        validation_files[i] = f"{audio_dir}/{spk}/{basename}.wav"
    
    random.seed(1234)
    random.shuffle(training_files)
    random.shuffle(validation_files)

    return training_files, validation_files


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, use_aug=False):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.use_aug = use_aug

        with open("filelists/spk2id.json") as f:
            self.spk2id = json.load(f)

    def __getitem__(self, index):
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_wav(filename)
            audio = audio / MAX_WAV_VALUE
            audio = normalize(audio) * 0.95
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        if audio.size(1) >= self.segment_size:
            max_audio_start = audio.size(1) - self.segment_size
            audio_start = random.randint(0, max_audio_start)
            audio = audio[:, audio_start:audio_start+self.segment_size]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        mel = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                center=False)

        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)
        
        spk_path = filename.replace("audio", "spk").replace(".wav", ".npy")
        spk_emb = torch.from_numpy(np.load(spk_path)) # (256)
        spk = filename.split("/")[-2]
        spk_id = self.spk2id[spk]
        spk_id = torch.LongTensor([spk_id])

        if not self.use_aug:
            return (mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze(), spk_emb, spk_id)
        
        mel_aug, _ = mel_spectogram(
            audio=audio.squeeze(),
            sample_rate=16000,
            hop_length=256,
            win_length=1024,
            n_mels=80,
            n_fft=1024,
            f_min=0.0,
            f_max=8000.0,
            power=1,
            normalized=False,
            min_max_energy_norm=True,
            norm="slaney",
            mel_scale="slaney",
            compression=True
        )
        mel_aug = self.resize_mel(mel_aug.unsqueeze(0)).squeeze(0)

        return (mel_aug.squeeze(), mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze(), spk_emb, spk_id)

    def __len__(self):
        return len(self.audio_files)
    
    def resize_mel(self, mel):
        ratio = 0.85 + 0.3 * torch.rand(1) # 0.85 ~ 1.15
        height = int(mel.size(-2) * ratio)
        width = mel.size(-1)
        
        mel_r = resize(mel, (height, width), antialias=True)
        
        if height >= mel.size(-2):
            mel_r = mel_r[:, :mel.size(-2), :]
        else:
            pad = mel_r[:, -1:, :].repeat(1, mel.size(-2) - height, 1) 
            pad += torch.randn_like(pad) / 1e3
            mel_r = torch.cat((mel_r, pad), 1)
        
        return mel_r
