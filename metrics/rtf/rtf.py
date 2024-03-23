import os
import argparse
import json
import math
import time

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


def get_emb_tgts(wav_tgt, embedding_models, feature_extractor):
    emb_tgts = []
    for embedding_model in embedding_models:
        emb_tgt = compute_embedding(wav_tgt, embedding_model, feature_extractor)
        emb_tgts.append(emb_tgt)
    return emb_tgts


def get_sim(y, emb_tgts, embedding_models, feature_extractor):
    similarity = 0
    for embedding_model, emb_tgt in zip(embedding_models, emb_tgts):
        similarity += compute_similarity2(y.squeeze(1), emb_tgt, embedding_model, feature_extractor)
    similarity /= len(embedding_models)
    return similarity


def get_best_wav(x, initial_f0, wav_tgt, generator, embedding_models, feature_extractor, search):
    y = generator.infer(x, initial_f0)
    if not search:
        return y
    
    step = (math.log(1100) - math.log(50)) / 256
    threshold = 10
    voiced = initial_f0 > threshold
    initial_lf0 = torch.log(initial_f0)
    
    emb_tgts = get_emb_tgts(wav_tgt, embedding_models, feature_extractor)
    best_similarity = get_sim(y, emb_tgts, embedding_models, feature_extractor)
    best_wav = y

    for search_direction in [1, -1]:
        search = True
        tolerance = 3
        i = 0
        
        while search:
            i += search_direction
            
            lf0 = initial_lf0 + step * i
            f0 = torch.exp(lf0)
            f0 = torch.where(voiced, f0, initial_f0)
            y = generator.infer(x, initial_f0)

            similarity = get_sim(y, emb_tgts, embedding_models, feature_extractor)

            if similarity > best_similarity:
                best_similarity = similarity
                best_wav = y
                tolerance = 3
            else:
                tolerance -= 1
            if tolerance == 0:
                search = False

    return best_wav


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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # load models
    F0_model = JDCNet(num_class=1, seq_len=192)
    generator = Generator(h, F0_model).to(device)

    state_dict_g = torch.load(args.ptfile, map_location=device)
    generator.load_state_dict(state_dict_g['generator'], strict=True)
    generator.remove_weight_norm()
    _ = generator.eval()

    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
    wavlm.eval()
    wavlm.to(device)

    asv_model_ids = [
        "damo/speech_eres2net_sv_zh-cn_16k-common",
        "damo/speech_eres2net_sv_en_voxceleb_16k"
    ]
    if args.search:
        embedding_models, feature_extractor = get_asv_models(asv_model_ids, args.asv_dir)
        for embedding_model in embedding_models:
            embedding_model.to(device)
    else:
        embedding_models = [None] * len(asv_model_ids)
        feature_extractor = None

    # load stats
    with open("filelists/spk2id.json") as f:
        spk2id = json.load(f)
    with open("filelists/f0_stats.json") as f:
        f0_stats = json.load(f)
    
    # load text
    with open(args.txtpath, "r") as f:
        lines = f.readlines()

    # synthesize
    total_rtf = 0
    cnt = 0
    with torch.no_grad():
        for line in tqdm(lines[:100]):
            title, src_wav, tgt_wav, tgt_spk, tgt_emb = line.strip().split("|")
            
            # tgt
            spk_id = spk2id[tgt_spk]
            spk_id = torch.LongTensor([spk_id]).unsqueeze(0).to(device)
            
            spk_emb = np.load(tgt_emb)
            spk_emb = torch.from_numpy(spk_emb).unsqueeze(0).to(device)

            f0_mean_tgt = f0_stats[tgt_spk]["mean"]
            f0_mean_tgt = torch.FloatTensor([f0_mean_tgt]).unsqueeze(0).to(device)

            wav_tgt, sr = librosa.load(tgt_wav, sr=16000)
            wav_tgt = torch.FloatTensor(wav_tgt).to(device)

            # src
            wav, sr = librosa.load(src_wav, sr=16000)
            wav = torch.FloatTensor(wav).to(device)
            mel = mel_spectrogram(wav.unsqueeze(0), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)
            
            length_y = wav.size(-1) / 16000
            start = time.time()

            x = wavlm(wav.unsqueeze(0)).last_hidden_state
            x = x.transpose(1, 2) # (B, C, T)
            x = F.pad(x, (0, mel.size(2) - x.size(2)), 'constant')

            # cvt
            f0 = generator.get_f0(mel, f0_mean_tgt)
            x = generator.get_x(x, spk_emb, spk_id)
            y = get_best_wav(x, f0, wav_tgt, generator, embedding_models, feature_extractor, search=args.search)
            
            rtf = (time.time() - start) / length_y
            total_rtf += rtf
            cnt += 1

            # audio = y.squeeze()
            # audio = audio / torch.max(torch.abs(audio)) * 0.95
            # audio = audio * MAX_WAV_VALUE
            # audio = audio.cpu().numpy().astype('int16')

            # output_file = os.path.join(args.outdir, f"{title}.wav")
            # sf.write(output_file, audio, h.sampling_rate, "PCM_16")
    print(f"RTF: {total_rtf / cnt:.6f}")
