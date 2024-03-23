import os
import argparse
import json
import math

import librosa
import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np
from tqdm import tqdm
from transformers import WavLMModel

from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE
from models import Generator
from Utils.JDC.model import JDCNet
from asv import compute_similarity2, compute_embedding, get_asv_models

from thop import profile, clever_format


class Model(torch.nn.Module):
    def __init__(self, generator, wavlm):
        super().__init__()
        self.generator = generator
        self.wavlm = wavlm

    def forward(self, wav, mel, f0_mean_tgt, spk_emb, spk_id):
        x = self.wavlm(wav.unsqueeze(0)).last_hidden_state
        x = x.transpose(1, 2) # (B, C, T)
        x = F.pad(x, (0, mel.size(2) - x.size(2)), 'constant')

        f0 = self.generator.get_f0(mel, f0_mean_tgt)
        x = self.generator.get_x(x, spk_emb, spk_id)
        y = self.generator.infer(x, f0)


class F0(torch.nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, mel, f0_mean_tgt):
        f0 = self.generator.get_f0(mel, f0_mean_tgt)


class Voc(torch.nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, x, f0):
        y = self.generator.infer(x, f0)


class Enc(torch.nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, x):
        x = self.generator.enc(x)


class Dec(torch.nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, x, g):
        x = self.generator.dec(x, g=g)


class Spk(torch.nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, spk_id, spk_emb):
        g = self.generator.embed_spk(spk_id).transpose(1, 2)
        g = g + spk_emb.unsqueeze(-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpfile", type=str, default="config_v1_16k.json", help="path to json config file")
    parser.add_argument("--ptfile", type=str, default="exp/default/g_00700000", help="path to pth file")
    parser.add_argument("--txtpath", type=str, default="test/txts/u2s.txt", help="path to txt file")
    parser.add_argument("--outdir", type=str, default="output/test", help="path to output dir")
    parser.add_argument("--search", default=False, action="store_true", help="search f0")
    parser.add_argument("--asv_dir", default="/home/lijingyi/code/3D-Speaker/pretrained", help="asv model checkpoints dir")
    args = parser.parse_args()
    
    # os.makedirs(args.outdir, exist_ok=True)

    # load config
    with open(args.hpfile) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    # global device
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(h.seed)
    #     device = torch.device('cuda')
    # else:
    #     device = torch.device('cpu')

    # load models
    F0_model = JDCNet(num_class=1, seq_len=192)
    generator = Generator(h, F0_model)#.to(device)

    # state_dict_g = torch.load(args.ptfile, map_location=device)
    # generator.load_state_dict(state_dict_g['generator'], strict=True)
    # generator.remove_weight_norm()
    # _ = generator.eval()

    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
    # wavlm.eval()
    # wavlm.to(device)

    # load stats
    with open("filelists/spk2id.json") as f:
        spk2id = json.load(f)
    with open("filelists/f0_stats.json") as f:
        f0_stats = json.load(f)
    
    # load text
    with open(args.txtpath, "r") as f:
        lines = f.readlines()

    # define model & modules
    model = Model(generator, wavlm)#.to(device)
    mf0 = F0(generator)#.to(device)
    mvoc = Voc(generator)#.to(device)
    menc = Enc(generator)#.to(device)
    mdec = Dec(generator)#.to(device)
    mspk = Spk(generator)#.to(device)

    # synthesize
    with torch.no_grad():
        line = lines[0]
        title, src_wav, tgt_wav, tgt_spk, tgt_emb = line.strip().split("|")
        
        # tgt
        spk_id = spk2id[tgt_spk]
        spk_id = torch.LongTensor([spk_id]).unsqueeze(0)#.to(device)
        
        spk_emb = np.load(tgt_emb)
        spk_emb = torch.from_numpy(spk_emb).unsqueeze(0)#.to(device)

        f0_mean_tgt = f0_stats[tgt_spk]["mean"]
        f0_mean_tgt = torch.FloatTensor([f0_mean_tgt]).unsqueeze(0)#.to(device)

        wav_tgt, sr = librosa.load(tgt_wav, sr=16000)
        wav_tgt = torch.FloatTensor(wav_tgt)#.to(device)

        # src
        wav, sr = librosa.load(src_wav, sr=16000)
        wav = torch.FloatTensor(wav)#.to(device)
        mel = mel_spectrogram(wav.unsqueeze(0), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)
        
        # macs: model
        print("--- model ---")
        macs, params = profile(model, inputs=(wav, mel, f0_mean_tgt, spk_emb, spk_id))
        macs, params = clever_format([macs, params], "%.3f")
        print(macs, params)

        # macs: f0
        print("--- f0 ---")
        macs, params = profile(mf0, inputs=(mel, f0_mean_tgt))
        macs, params = clever_format([macs, params], "%.3f")
        print(macs, params)

        # macs: wavlm
        print("--- wavlm ---")
        macs, params = profile(wavlm, inputs=(wav.unsqueeze(0),))
        macs, params = clever_format([macs, params], "%.3f")
        print(macs, params)

        # macs: spk
        print("--- spk ---")
        macs, params = profile(mspk, inputs=(spk_id, spk_emb))
        macs, params = clever_format([macs, params], "%.3f")
        print(macs, params)
            
        # macs: enc
        print("--- enc ---")
        x = wavlm(wav.unsqueeze(0)).last_hidden_state
        x = x.transpose(1, 2) # (B, C, T)
        x = F.pad(x, (0, mel.size(2) - x.size(2)), 'constant')
        x_enc = x

        macs, params = profile(menc, inputs=(x,))
        macs, params = clever_format([macs, params], "%.3f")
        print(macs, params)

        # macs: dec
        print("--- dec ---")
        x = generator.enc(x)
        g = generator.embed_spk(spk_id).transpose(1, 2)
        g = g + spk_emb.unsqueeze(-1)

        macs, params = profile(mdec, inputs=(x, g))
        macs, params = clever_format([macs, params], "%.3f")
        print(macs, params)

        # macs: voc
        print("--- voc ---")
        f0 = generator.get_f0(mel, f0_mean_tgt)
        x = generator.get_x(x_enc, spk_emb, spk_id)

        macs, params = profile(mvoc, inputs=(x, f0))
        macs, params = clever_format([macs, params], "%.3f")
        print(macs, params)

        # cvt
        # f0 = generator.get_f0(mel, f0_mean_tgt)
        # x = generator.get_x(x, spk_emb, spk_id)
        # y = get_best_wav(x, f0, wav_tgt, generator, embedding_models, feature_extractor, search=args.search)
        
        # audio = y.squeeze()
        # audio = audio / torch.max(torch.abs(audio)) * 0.95
        # audio = audio * MAX_WAV_VALUE
        # audio = audio.cpu().numpy().astype('int16')

        # output_file = os.path.join(args.outdir, f"{title}.wav")
        # sf.write(output_file, audio, h.sampling_rate, "PCM_16")
