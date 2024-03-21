import os
import argparse

from tqdm import tqdm
from glob import glob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_dir", type=str, default="CONVERTED")
    parser.add_argument("--tgt_dir", type=str, default="TEST_/tgt")
    parser.add_argument("--title", type=str, default="samples")
    args = parser.parse_args()
    
    wavs = glob(f'{args.wav_dir}/*.wav', recursive=True)

    os.makedirs("txts", exist_ok=True)
    with open(f"txts/{args.title}.txt", "w") as f:
        for wav in tqdm(wavs):
            names = wav.split("-")
            src, tgt = names
            f.write(f"{wav}|{args.tgt_dir}/{tgt}\n")
