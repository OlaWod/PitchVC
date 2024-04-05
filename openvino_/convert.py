
import os
import argparse
import json

import openvino as ov
import librosa
import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np
from tqdm import tqdm

from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE
from stft import TorchSTFT


def infer(wav, mel, spk_emb, spk_id, f0_mean_tgt):
    # g1
    out = g1([wav, mel, spk_emb, spk_id, f0_mean_tgt])
    x = out[g1.output(0)]
    har_source = out[g1.output(1)]

    # stft
    har_source = torch.from_numpy(har_source)
    har_spec, har_phase = stft.transform(har_source)
    har_spec, har_phase = har_spec.numpy(), har_phase.numpy()

    # g2
    out = g2([x, har_spec, har_phase])
    spec = out[g2.output(0)]
    phase = out[g2.output(1)]

    # istft
    spec, phase = torch.from_numpy(spec), torch.from_numpy(phase)
    y = stft.inverse(spec, phase)

    return y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpfile", type=str, default="config_v1_16k.json", help="path to json config file")
    parser.add_argument("--g1path", type=str, default="exp/g1.xml", help="path to g1 xml file")
    parser.add_argument("--g2path", type=str, default="exp/g2.xml", help="path to g2 xml file")
    parser.add_argument("--txtpath", type=str, default="test/txts/u2s.txt", help="path to txt file")
    parser.add_argument("--outdir", type=str, default="output/test", help="path to output dir")
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)

    # load config
    with open(args.hpfile) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    # load models
    core = ov.Core()
    g1 = core.read_model(model=args.g1path)
    g1 = core.compile_model(model=g1, device_name="CPU")
    g2 = core.read_model(model=args.g2path)
    g2 = core.compile_model(model=g2, device_name="CPU")

    stft = TorchSTFT(filter_length=h.gen_istft_n_fft, hop_length=h.gen_istft_hop_size, win_length=h.gen_istft_n_fft)

    # load stats
    with open("filelists/spk2id.json") as f:
        spk2id = json.load(f)
    with open("filelists/f0_stats.json") as f:
        f0_stats = json.load(f)
    
    # load text
    with open(args.txtpath, "r") as f:
        lines = f.readlines()

    # synthesize
    for line in tqdm(lines):
        title, src_wav, tgt_wav, tgt_spk, tgt_emb = line.strip().split("|")
        
        # tgt
        spk_id = spk2id[tgt_spk]
        spk_id = np.array([spk_id], dtype=np.int64)[None, :]
        
        spk_emb = np.load(tgt_emb)[None, :]

        f0_mean_tgt = f0_stats[tgt_spk]["mean"]
        f0_mean_tgt = np.array([f0_mean_tgt], dtype=np.float32)[None, :]

        # src
        wav, sr = librosa.load(src_wav, sr=16000)
        wav = wav[None, :]
        mel = mel_spectrogram(torch.from_numpy(wav), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax).numpy()
        
        # cvt
        y = infer(wav, mel, spk_emb, spk_id, f0_mean_tgt)
        
        audio = y.squeeze()
        audio = audio / torch.max(torch.abs(audio)) * 0.95
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype('int16')

        output_file = os.path.join(args.outdir, f"{title}.wav")
        sf.write(output_file, audio, h.sampling_rate, "PCM_16")
