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


def process_one(line, encoder, similarity, title):
    src, tgt = line.strip().split("|")
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
    # ssims[basename] = ssim
    with open(f"tmp/{title}+{basename}.txt", "w") as f:
        f.write(f"{ssim}")


def process_batch(batch, title):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load models
    encoder = VoiceEncoder()
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    # process
    rank = mp.current_process()._identity
    rank = rank[0] if len(rank) > 0 else 0
    for line in tqdm(batch, position=rank):
        process_one(line, encoder, similarity, title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--txtpath", type=str, default="samples.txt", help="path to txt file")
    parser.add_argument("--title", type=str, default="1", help="output title")
    parser.add_argument("--n_processes", type=int, default=8, help="number of multiprocessing processes")
    args = parser.parse_args()
    
    with open(args.txtpath, "r") as f:
        lines = f.readlines()

    # process
    shutil.rmtree("tmp", ignore_errors=True)
    os.makedirs("tmp", exist_ok=True)
    mp.set_start_method("spawn", force=True)
    n_processes = args.n_processes
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        tasks = []
        for i in range(n_processes):
            start = int(i * len(lines) / n_processes)
            end = int((i + 1) * len(lines) / n_processes)
            batch = lines[start:end]
            tasks.append(executor.submit(process_batch, batch, args.title))
        for task in tqdm(tasks, position = 0):
            task.result()

    # write results
    ssims = {}
    for txt in glob(f"tmp/{args.title}+*.txt"):
        basename = txt.split("/")[-1].split(".")[0]
        basename = basename.split("+")[-1]
        with open(txt, "r") as f:
            ssim = float(f.readline().strip())
        ssims[basename] = ssim

    os.makedirs("result", exist_ok=True)
    with open(f"result/{args.title}.json", "w") as f:
        json.dump(ssims, f, indent=2)
    with open(f"result/{args.title}.txt", "w") as f:
        ssims = list(ssims.values())
        ssim = sum(ssims) / len(ssims)
        f.write(f"mean: {ssim}")
    print("mean: ", ssim)
