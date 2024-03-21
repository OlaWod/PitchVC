import os
import shutil
import json
from tqdm import tqdm
from glob import glob
import numpy as np
import pyworld as pw
import argparse
import torch
import librosa
import parselmouth
from torchmetrics import PearsonCorrCoef
from concurrent.futures import ProcessPoolExecutor


def get_f0(x, fs=16000, n_shift=160):
    x = x.astype(np.float64)
    frame_period = n_shift / fs * 1000
    f0, timeaxis = pw.dio(x, fs, frame_period=frame_period)
    f0 = pw.stonemask(x, f0, timeaxis, fs)
    return f0


def compute_f0(wav, sr=16000, frame_period=10.0):
    wav = wav.astype(np.float64)
    f0, timeaxis = pw.harvest(
        wav, sr, frame_period=frame_period, f0_floor=20.0, f0_ceil=600.0)
    return f0


def process_one(src, title, pearson):
    src_basename = src.split("/")[-1].split(".")[0]
    src = librosa.load(src, sr=16000)[0]
    src_f0 = get_f0(src)
    src_f0 = torch.from_numpy(src_f0)
    # src_f0 = torch.where(src_f0 > 0, torch.log(src_f0), torch.zeros_like(src_f0))

    for cvt in glob(f'{args.wavdir}/{src_basename}-*.wav'):
        cvt_basename = cvt.split("/")[-1].split(".")[0]
        cvt = librosa.load(cvt, sr=16000)[0]
        cvt_f0 = get_f0(cvt)
        if sum(cvt_f0) == 0:
            cvt_f0 = compute_f0(cvt)
            print(cvt_basename)
        cvt_f0 = torch.from_numpy(cvt_f0)
        # cvt_f0 = torch.where(cvt_f0 > 0, torch.log(cvt_f0), torch.zeros_like(cvt_f0))

        # assert abs(src_f0.shape[-1] - cvt_f0.shape[-1]) < 3, (src_f0.shape, cvt_f0.shape)

        pcc = pearson(src_f0[:cvt_f0.shape[-1]], cvt_f0[:src_f0.shape[-1]]).item()
        with open(f"tmp/{title}+{cvt_basename}.txt", "w") as f:
            f.write(f"{pcc}")


def process_batch(file_chunk, title):
    pearson = PearsonCorrCoef()

    # process
    for line in tqdm(file_chunk):
        process_one(line, title, pearson)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavdir", type=str, default="PROPOSED")
    parser.add_argument("--srcdir", type=str, default="SRC_VCTK")
    parser.add_argument("--title", type=str, default="1", help="output title")
    parser.add_argument("--n_processes", type=int, default=32, help="number of multiprocessing processes")
    args = parser.parse_args()

    srcs = glob(f'{args.srcdir}/*.wav')    

    # process
    shutil.rmtree("tmp", ignore_errors=True)
    os.makedirs("tmp", exist_ok=True)
    n_processes = args.n_processes
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        tasks = []
        for i in range(n_processes):
            start = int(i * len(srcs) / n_processes)
            end = int((i + 1) * len(srcs) / n_processes)
            batch = srcs[start:end]
            tasks.append(executor.submit(process_batch, batch, args.title))
        for task in tqdm(tasks, position=0):
            task.result()

    # write results
    pccs = {}
    for txt in glob(f"tmp/{args.title}+*.txt"):
        basename = txt.split("/")[-1].split(".")[0]
        basename = basename.split("+")[-1]
        with open(txt, "r") as f:
            pcc = float(f.readline().strip())
        pccs[basename] = pcc
            
    os.makedirs("result", exist_ok=True)
    with open(f"result/{args.title}.json", "w") as f:
        json.dump(pccs, f, indent=4)
    with open(f"result/{args.title}.txt", "w") as f:
        pcc = sum(list(pccs.values())) / len(pccs)
        f.write(f"mean: {pcc}")
    print("mean: ", pcc)
