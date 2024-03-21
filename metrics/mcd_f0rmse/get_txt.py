import os
import json
import argparse
import random

from tqdm import tqdm
from glob import glob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_dir", type=str, default="CONVERTED")
    # parser.add_argument("--tgt_dir", type=str, default="TGT")
    parser.add_argument("--title", type=str, default="samples")
    args = parser.parse_args()

    with open("txts/valid.json", "r") as f:
        valid = json.load(f)
    
    wavs = glob(f'{args.wav_dir}/*.wav', recursive=True)

    lines = []
    for wav in tqdm(wavs):
        names = wav.split("-")
        src, tgt = names
        src = src.split("/")[-1]
        if src not in valid:
            continue
        tgt_spk, tgt = valid[src].split("|")
        lines.append(f"{wav}|TEST_/VCTK/{tgt_spk}/{tgt}.wav\n")

    os.makedirs("txts", exist_ok=True)
    with open(f"txts/{args.title}.txt", "w") as f:
        for line in lines:
            f.write(line)
