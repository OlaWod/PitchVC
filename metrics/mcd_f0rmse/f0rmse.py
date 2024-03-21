# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import shutil

import librosa
import torch
import numpy as np

from utils import JsonHParams, get_f0_features_using_parselmouth, get_pitch_sub_median


ZERO = 1e-8


def extract_f0rmse(
    audio_ref,
    audio_deg,
    fs=None,
    hop_length=256,
    f0_min=37,
    f0_max=1000,
    pitch_bin=256,
    pitch_max=1100.0,
    pitch_min=50.0,
    need_mean=True,
    method="dtw",
):
    """Compute F0 Root Mean Square Error (RMSE) between the predicted and the ground truth audio.
    audio_ref: path to the ground truth audio.
    audio_deg: path to the predicted audio.
    fs: sampling rate.
    hop_length: hop length.
    f0_min: lower limit for f0.
    f0_max: upper limit for f0.
    pitch_bin: number of bins for f0 quantization.
    pitch_max: upper limit for f0 quantization.
    pitch_min: lower limit for f0 quantization.
    need_mean: subtract the mean value from f0 if "True".
    method: "dtw" will use dtw algorithm to align the length of the ground truth and predicted audio.
            "cut" will cut both audios into a same length according to the one with the shorter length.
    """
    # Load audio
    if fs != None:
        audio_ref, _ = librosa.load(audio_ref, sr=fs)
        audio_deg, _ = librosa.load(audio_deg, sr=fs)
    else:
        audio_ref, fs = librosa.load(audio_ref)
        audio_deg, fs = librosa.load(audio_deg)

    # Initialize config for f0 extraction
    cfg = JsonHParams()
    cfg.sample_rate = fs
    cfg.hop_size = hop_length
    cfg.f0_min = f0_min
    cfg.f0_max = f0_max
    cfg.pitch_bin = pitch_bin
    cfg.pitch_max = pitch_max
    cfg.pitch_min = pitch_min

    # Extract f0
    f0_ref = get_f0_features_using_parselmouth(
        audio_ref,
        cfg,
    )[0]

    f0_deg = get_f0_features_using_parselmouth(
        audio_deg,
        cfg,
    )[0]

    # Subtract mean value from f0
    if need_mean:
        f0_ref = torch.from_numpy(f0_ref)
        f0_deg = torch.from_numpy(f0_deg)

        f0_ref = get_pitch_sub_median(f0_ref).numpy()
        f0_deg = get_pitch_sub_median(f0_deg).numpy()

    # Avoid silence
    min_length = min(len(f0_ref), len(f0_deg))
    if min_length <= 1:
        return 0

    # F0 length alignment
    if method == "cut":
        length = min(len(f0_ref), len(f0_deg))
        f0_ref = f0_ref[:length]
        f0_deg = f0_deg[:length]
    elif method == "dtw":
        _, wp = librosa.sequence.dtw(f0_ref, f0_deg, backtrack=True)
        f0_gt_new = []
        f0_pred_new = []
        for i in range(wp.shape[0]):
            gt_index = wp[i][0]
            pred_index = wp[i][1]
            f0_gt_new.append(f0_ref[gt_index])
            f0_pred_new.append(f0_deg[pred_index])
        f0_ref = np.array(f0_gt_new)
        f0_deg = np.array(f0_pred_new)
        assert len(f0_ref) == len(f0_deg)

    # Compute RMSE
    f0_mse = np.square(np.subtract(f0_ref, f0_deg)).mean()
    f0_rmse = math.sqrt(f0_mse)

    return f0_rmse


import argparse
import json
import os

import torch
import numpy as np
from tqdm import tqdm
from glob import glob
from resemblyzer import VoiceEncoder, preprocess_wav
from concurrent.futures import ProcessPoolExecutor
import torch.multiprocessing as mp


def process_one(line, title):
    src, tgt = line.strip().split("|")
    if not os.path.exists(tgt):
        return
    basename = src.split("/")[-1].split(".")[0]

    mcd = extract_f0rmse(src, tgt, 16000)

    # f0rmses[basename] = f0rmse
    with open(f"tmp/{title}+{basename}.txt", "w") as f:
        f.write(f"{mcd}")


def process_batch(batch, title):
    # process
    for line in tqdm(batch):
        process_one(line, title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--txtpath", type=str, default="samples.txt", help="path to txt file")
    parser.add_argument("--title", type=str, default="1", help="output title")
    parser.add_argument("--n_processes", type=int, default=16, help="number of multiprocessing processes")
    args = parser.parse_args()
    
    with open(args.txtpath, "r") as f:
        lines = f.readlines()

    # process
    shutil.rmtree("tmp", ignore_errors=True)
    os.makedirs("tmp", exist_ok=True)
    n_processes = args.n_processes
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        tasks = []
        for i in range(n_processes):
            start = int(i * len(lines) / n_processes)
            end = int((i + 1) * len(lines) / n_processes)
            batch = lines[start:end]
            tasks.append(executor.submit(process_batch, batch, args.title))
        for task in tqdm(tasks, position=0):
            task.result()

    # write results
    f0rmses = {}
    for txt in glob(f"tmp/{args.title}+*.txt"):
        basename = txt.split("/")[-1].split(".")[0]
        basename = basename.split("+")[-1]
        with open(txt, "r") as f:
            f0rmse = float(f.readline().strip())
        f0rmses[basename] = f0rmse

    os.makedirs("result", exist_ok=True)
    with open(f"result/{args.title}.json", "w") as f:
        json.dump(f0rmses, f, indent=2)
    with open(f"result/{args.title}.txt", "w") as f:
        f0rmses = list(f0rmses.values())
        f0rmse = sum(f0rmses) / len(f0rmses)
        f.write(f"mean: {f0rmse}")
    print("mean: ", f0rmse)
