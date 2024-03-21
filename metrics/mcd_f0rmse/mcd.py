# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pymcd.mcd import Calculate_MCD


def extract_mcd(audio_ref, audio_deg, fs=None, mode="dtw_sl"):
    """Extract Mel-Cepstral Distance for a two given audio.
    Args:
        audio_ref: The given reference audio. It is an audio path.
        audio_deg: The given synthesized audio. It is an audio path.
        mode: "plain", "dtw" and "dtw_sl".
    """
    mcd_toolbox = Calculate_MCD(MCD_mode=mode)
    if fs != None:
        mcd_toolbox.SAMPLING_RATE = fs
    mcd_value = mcd_toolbox.calculate_mcd(audio_ref, audio_deg)

    return mcd_value


import argparse
import json
import os
import shutil

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

    mcd = extract_mcd(src, tgt, 16000)

    # mcds[basename] = mcd
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
    mcds = {}
    for txt in glob(f"tmp/{args.title}+*.txt"):
        basename = txt.split("/")[-1].split(".")[0]
        basename = basename.split("+")[-1]
        with open(txt, "r") as f:
            mcd = float(f.readline().strip())
        mcds[basename] = mcd

    os.makedirs("result", exist_ok=True)
    with open(f"result/{args.title}.json", "w") as f:
        json.dump(mcds, f, indent=2)
    with open(f"result/{args.title}.txt", "w") as f:
        mcds = list(mcds.values())
        mcd = sum(mcds) / len(mcds)
        f.write(f"mean: {mcd}")
    print("mean: ", mcd)
