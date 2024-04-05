
import os
import argparse
import json

import onnxruntime
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
    ort_inputs = {
        ort_session_g1.get_inputs()[0].name: wav,
        ort_session_g1.get_inputs()[1].name: mel,
        ort_session_g1.get_inputs()[2].name: spk_emb,
        ort_session_g1.get_inputs()[3].name: spk_id,
        ort_session_g1.get_inputs()[4].name: f0_mean_tgt,
    }
    ort_outs = ort_session_g1.run(None, ort_inputs)
    x, har_source = ort_outs[0], ort_outs[1]

    # stft
    har_source = torch.from_numpy(har_source)
    har_spec, har_phase = stft.transform(har_source)
    har_spec, har_phase = har_spec.numpy(), har_phase.numpy()

    # g2
    ort_inputs = {
        ort_session_g2.get_inputs()[0].name: x,
        ort_session_g2.get_inputs()[1].name: har_spec,
        ort_session_g2.get_inputs()[2].name: har_phase,
    }
    ort_outs = ort_session_g2.run(None, ort_inputs)
    spec, phase = ort_outs[0], ort_outs[1]

    # istft
    spec, phase = torch.from_numpy(spec), torch.from_numpy(phase)
    y = stft.inverse(spec, phase)

    return y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpfile", type=str, default="config_v1_16k.json", help="path to json config file")
    parser.add_argument("--g1path", type=str, default="exp/g1.onnx", help="path to g1 onnx file")
    parser.add_argument("--g2path", type=str, default="exp/g2.onnx", help="path to g2 onnx file")
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
    ort_session_g1 = onnxruntime.InferenceSession(args.g1path, providers=["CPUExecutionProvider"])
    ort_session_g2 = onnxruntime.InferenceSession(args.g2path, providers=["CPUExecutionProvider"])

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
