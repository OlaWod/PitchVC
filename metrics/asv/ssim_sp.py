import argparse
import json
import os

import torch
import numpy as np
from tqdm import tqdm
from resemblyzer import VoiceEncoder, preprocess_wav


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--txtpath", type=str, default="samples.txt", help="path to txt file")
    parser.add_argument("--title", type=str, default="1", help="output title")
    args = parser.parse_args()
    
    encoder = VoiceEncoder()
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    
    ssims = {}
    with open(args.txtpath, "r") as f:
        for rawline in tqdm(f.readlines()):
            src, tgt = rawline.strip().split("|")
            basename = src.split("/")[-1].split(".")[0]

            src = preprocess_wav(src)
            src = src / np.max(np.abs(src)) * 0.98
            src = encoder.embed_utterance(src)
            src = torch.from_numpy(src)

            tgt = preprocess_wav(tgt)
            tgt = tgt / np.max(np.abs(tgt)) * 0.98
            tgt = encoder.embed_utterance(tgt)
            tgt = torch.from_numpy(tgt)

            ssim = similarity(src, tgt).item()
            # ssim = np.inner(src, tgt)
            # ssim /= (np.linalg.norm(src) * np.linalg.norm(tgt))
            ssims[basename] = ssim
            
    os.makedirs("result", exist_ok=True)
    with open(f"result/{args.title}.json", "w") as f:
        json.dump(ssims, f, indent=2)
    with open(f"result/{args.title}.txt", "w") as f:
        ssims = list(ssims.values())
        ssim = sum(ssims) / len(ssims)
        f.write(f"mean: {ssim}")
    print("mean: ", ssim)
