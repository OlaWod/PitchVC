import os
import random
import shutil
from glob import glob


# vctk
uttr_n = 300
src_dir = "dataset/vctk-16k"
shutil.rmtree("test/TEST_SRC_VCTK", ignore_errors=True)
os.makedirs("test/TEST_SRC_VCTK", exist_ok=True)

wavpaths = []
with open("filelists/test.txt", "r") as f:
    for line in f.readlines():
        spk, basename = line.strip().split("|")
        wavpaths.append(f"{src_dir}/{spk}/{basename}.wav")
random.shuffle(wavpaths)
wavpaths = random.sample(wavpaths, uttr_n)

for i, wavpath in enumerate(wavpaths):
    wavname = wavpath.split("/")[-1]
    shutil.copy(wavpath, f"test/TEST_SRC_VCTK/{wavname}")

# libri
uttr_n = 300
src_dir = "/home/Datasets/lijingyi/data/libri/test-clean"
shutil.rmtree("test/TEST_SRC_LRBRI", ignore_errors=True)
os.makedirs("test/TEST_SRC_LRBRI", exist_ok=True)

wavpaths = glob(f'{src_dir}/*/*/*.wav', recursive=True)
random.shuffle(wavpaths)
wavpaths = random.sample(wavpaths, uttr_n)

for i, wavpath in enumerate(wavpaths):
    wavname = wavpath.split("/")[-1]
    shutil.copy(wavpath, f"test/TEST_SRC_LRBRI/{wavname}")

# esd_en
uttr_n = 300
src_dir = "/home/Datasets/lijingyi/data/esd/Emotional Speech Dataset (ESD)"
shutil.rmtree("test/TEST_SRC_ESD_EN", ignore_errors=True)
os.makedirs("test/TEST_SRC_ESD_EN", exist_ok=True)

wavpaths = []
for spk in os.listdir(src_dir):
    if not os.path.isdir(f"{src_dir}/{spk}"):
        continue
    if int(spk) <= 10: # only use 0011~0020
        continue
    for emo in os.listdir(f"{src_dir}/{spk}"):
        if not os.path.isdir(f"{src_dir}/{spk}/{emo}"):
            continue
        if emo == "Neutral":
            continue
        wavs = glob(f"{src_dir}/{spk}/{emo}/*/*.wav", recursive=True)
        wavpaths.extend(wavs)
random.shuffle(wavpaths)
wavpaths = random.sample(wavpaths, uttr_n)

for i, wavpath in enumerate(wavpaths):
    wavname = wavpath.split("/")[-1]
    shutil.copy(wavpath, f"test/TEST_SRC_ESD_EN/{wavname}")

# esd_zh
uttr_n = 300
src_dir = "/home/Datasets/lijingyi/data/esd/Emotional Speech Dataset (ESD)"
shutil.rmtree("test/TEST_SRC_ESD_ZH", ignore_errors=True)
os.makedirs("test/TEST_SRC_ESD_ZH", exist_ok=True)

wavpaths = []
for spk in os.listdir(src_dir):
    if not os.path.isdir(f"{src_dir}/{spk}"):
        continue
    if int(spk) > 10: # only use 0001~0010
        continue
    for emo in os.listdir(f"{src_dir}/{spk}"):
        if not os.path.isdir(f"{src_dir}/{spk}/{emo}"):
            continue
        if emo != "Neutral":
            continue
        wavs = glob(f"{src_dir}/{spk}/{emo}/*/*.wav", recursive=True)
        wavpaths.extend(wavs)
random.shuffle(wavpaths)
wavpaths = random.sample(wavpaths, uttr_n)

for i, wavpath in enumerate(wavpaths):
    wavname = wavpath.split("/")[-1]
    shutil.copy(wavpath, f"test/TEST_SRC_ESD_ZH/{wavname}")
