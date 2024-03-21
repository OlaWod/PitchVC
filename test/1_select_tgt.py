import os
import json
import collections
import random
import shutil
from glob import glob

spk_n = 6
src_dir = "dataset/vctk-16k"

with open("filelists/spk_stats.json", "r") as f:
    spk_stats = json.load(f)

F, M = [], []
with open("test/spk_gender/speaker-info.txt", "r") as f:
    for line in f.readlines()[1:-1]:
        spk, _, gender = line.strip().split()[:3]
        if spk not in spk_stats:
            continue
        if gender == "F":
            F.append(spk)
        else:
            M.append(spk)

fs = random.sample(F, spk_n)
ms = random.sample(M, spk_n)

shutil.rmtree("test/TEST_TGT", ignore_errors=True)
os.makedirs("test/TEST_TGT", exist_ok=True)

for spk in fs:
    basename = spk_stats[spk]["best_spk_emb"]
    wav = f"{src_dir}/{spk}/{basename}.wav"
    shutil.copy(wav, f"test/TEST_TGT/F_{basename}.wav")
for spk in ms:
    basename = spk_stats[spk]["best_spk_emb"]
    wav = f"{src_dir}/{spk}/{basename}.wav"
    shutil.copy(wav, f"test/TEST_TGT/M_{basename}.wav")