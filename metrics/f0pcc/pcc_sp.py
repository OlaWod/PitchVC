import os
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


def get_f0(x, fs=16000, n_shift=160):
    x = x.astype(np.float64)
    frame_period = n_shift / fs * 1000
    f0, timeaxis = pw.dio(x, fs, frame_period=frame_period)
    f0 = pw.stonemask(x, f0, timeaxis, fs)
    return f0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavdir", type=str, default="PROPOSED")
    parser.add_argument("--srcdir", type=str, default="SRC_VCTK")
    parser.add_argument("--title", type=str, default="1", help="output title")
    args = parser.parse_args()

    pearson = PearsonCorrCoef()

    srcs = glob(f'{args.srcdir}/*.wav')    
    pccs = {}

    for src in tqdm(srcs):
        src_basename = src.split("/")[-1].split(".")[0]
        src = librosa.load(src, sr=16000)[0]
        src_f0 = get_f0(src)
        src_f0 = torch.from_numpy(src_f0)

        for cvt in glob(f'{args.wavdir}/{src_basename}-*.wav'):
            cvt_basename = cvt.split("/")[-1].split(".")[0]
            cvt = librosa.load(cvt, sr=16000)[0]
            cvt_f0 = get_f0(cvt)
            cvt_f0 = torch.from_numpy(cvt_f0)

            assert abs(src_f0.shape[-1] - cvt_f0.shape[-1]) < 2, (src_f0.shape, cvt_f0.shape)

            pcc = pearson(src_f0[:cvt_f0.shape[-1]], cvt_f0[:src_f0.shape[-1]]).item()
            # pcc = np.corrcoef(src_f0[:cvt_f0.shape[-1]], cvt_f0[:src_f0.shape[-1]])[0, 1]
            pccs[cvt_basename] = pcc
    
    os.makedirs("result", exist_ok=True)
    with open(f"result/{args.title}.json", "w") as f:
        json.dump(pccs, f, indent=4)
    with open(f"result/{args.title}.txt", "w") as f:
        pcc = sum(list(pccs.values())) / len(pccs)
        f.write(f"mean: {pcc}")
    print("mean: ", pcc)
