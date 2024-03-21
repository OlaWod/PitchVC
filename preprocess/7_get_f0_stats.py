import os
import json

import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


def remove_outlier(values):
    values = np.array(values)
    p25 = np.percentile(values, 25)
    p75 = np.percentile(values, 75)
    lower = p25 - 1.5 * (p75 - p25)
    upper = p75 + 1.5 * (p75 - p25)
    normal_indices = np.logical_and(values > lower, values < upper)

    return values[normal_indices]


root_dir = "dataset/f0"
threshold = 2
max_value = np.finfo(np.float64).min
min_value = np.finfo(np.float64).max
f0_stats = {}


for spk in tqdm(os.listdir(root_dir)):
    scaler = StandardScaler()
    f0_max = max_value
    f0_min = min_value
    for f0_path in glob(f"{root_dir}/{spk}/*.pt"):
        f0 = torch.load(f0_path)
        voiced = f0 > threshold
        f0 = f0[voiced]
        f0 = f0.numpy()
        f0 = remove_outlier(f0)
        if len(f0) > 0:
            scaler.partial_fit(f0.reshape((-1, 1)))
            f0_max = max(f0_max, f0.max())
            f0_min = min(f0_min, f0.min())
    f0_mean = scaler.mean_[0]
    f0_std = scaler.scale_[0]    
    f0_stats[spk] = {
        "mean": float(f0_mean),
        "std": float(f0_std),
        "max": float(f0_max),
        "min": float(f0_min),
    }

with open("filelists/f0_stats.json", "w") as f:
    json.dump(f0_stats, f, indent=2)
