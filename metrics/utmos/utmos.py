import argparse
import json
import os

import torch
import torchaudio
from glob import glob
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavdir", type=str, default="1", help="path to wav dir")
    parser.add_argument("--title", type=str, default="1", help="output title")
    parser.add_argument("--n_processes", type=int, default=8, help="number of multiprocessing processes")
    args = parser.parse_args()

    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).cuda()
    wavpaths = glob(f"{args.wavdir}/*.wav")

    results = {}
    for wavpath in tqdm(wavpaths):
        basename = wavpath.split("/")[-1].split(".")[0]
        wave, sr = torchaudio.load(wavpath)
        wave = wave.cuda()
        score = predictor(wave, sr)
        results[basename] = score.item()

    os.makedirs("result", exist_ok=True)
    with open(f"result/{args.title}.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(f"result/{args.title}.txt", "w") as f:
        results = list(results.values())
        result = sum(results) / len(results)
        f.write(f"mean: {result}")
    print("mean: ", result)
