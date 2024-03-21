import os

import torch
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
from glob import glob
from tqdm import tqdm

from Utils.JDC.model import JDCNet


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


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


F0_model = JDCNet(num_class=1, seq_len=192)
params = torch.load("Utils/JDC/bst.t7")['model']
F0_model.load_state_dict(params)
F0_model.cuda()

n_fft = 1024
num_mels = 80                                 
sampling_rate = 16000 
hop_size = 320
win_size = 1024
fmin = 0
fmax = 8000

wavs = glob(f"./dataset/vctk-16k/**/*.wav", recursive=True)
save_root = "./dataset/f0"

with torch.no_grad():
    for wav in tqdm(wavs):
        audio, sr = load_wav(wav)
        assert sr == sampling_rate
        audio = torch.FloatTensor(audio).cuda()
        audio = audio.unsqueeze(0)

        mel = mel_spectrogram(audio, n_fft, num_mels,
                                        sampling_rate, hop_size, win_size, fmin, fmax, center=False)
        f0, _, _ = F0_model(mel.unsqueeze(1))

        basename = wav.split("/")[-1].split(".")[0]
        spk = wav.split("/")[-2]
        save_dir = f"{save_root}/{spk}"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(f0.cpu(), f"{save_dir}/{basename}.pt")
