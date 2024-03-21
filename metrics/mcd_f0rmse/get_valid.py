import os
import json
import argparse
import random

from tqdm import tqdm
from glob import glob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_dir", type=str, default="CONVERTED")
    args = parser.parse_args()

    with open("txts/txt_map.json", "r") as f:
        txt_map = json.load(f)

    with open("txts/txt_map2.json", "r") as f:
        txt_map2 = json.load(f)
    
    wavs = glob(f'{args.wav_dir}/*.wav', recursive=True)

    valid = {}
    for wav in tqdm(wavs):
        names = wav.split("-")
        src, tgt = names
        src = src.split("/")[-1]
        tgt_spk = tgt.split("_")[1]
        txt = txt_map[src]
        spk_candidates = txt_map2[txt]
        if tgt_spk not in spk_candidates:
            continue
        tgt = spk_candidates[tgt_spk]
        valid[src] = f"{tgt_spk}|{tgt}"
    # valid = [(k, v) for k, v in valid.items()]
    # random.shuffle(valid)
    # valid = valid[:500]
    # valid = dict(valid)

    os.makedirs("txts", exist_ok=True)
    with open(f"txts/valid.json", "w") as f:
        json.dump(valid, f, indent=4)
