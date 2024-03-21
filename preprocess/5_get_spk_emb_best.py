import os
import json

import torch
import numpy as np
from tqdm import tqdm
from glob import glob


root_dir = "dataset/spk"
similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
spk_stats = {}

for spk in tqdm(os.listdir(root_dir)):
    spk_emb_all = 0
    n = 0
    for spk_path in glob(f"{root_dir}/{spk}/*.npy"):
        spk_emb = np.load(spk_path)
        spk_emb_all += spk_emb
        n += 1
    spk_emb_mean = spk_emb_all / n
    spk_emb_mean = torch.from_numpy(spk_emb_mean)

    max_score = 0
    best_spk_emb = None
    for spk_path in glob(f"{root_dir}/{spk}/*.npy"):
        spk_emb = np.load(spk_path)
        spk_emb = torch.from_numpy(spk_emb)
        score = similarity(spk_emb, spk_emb_mean).item()
        if score > max_score:
            max_score = score
            basename = spk_path.split("/")[-1].split(".")[0]
            best_spk_emb = basename

    spk_stats[spk] = {
        "best_spk_emb": best_spk_emb,
        "max_score": max_score,
    }

with open("filelists/spk_stats.json", "w") as f:
    json.dump(spk_stats, f, indent=2)
