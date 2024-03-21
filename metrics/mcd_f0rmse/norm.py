import os

import torch
import torchaudio
from glob import glob
from tqdm import tqdm


wavpaths = glob("TEST_/VCTK/*/*.wav")
for wavpath in tqdm(wavpaths):
    audio, sr = torchaudio.load(wavpath)
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    audio = audio / torch.max(torch.abs(audio)) * 0.95
    wavpath = wavpath.replace("VCTK", "VCTK_norm")
    wavdir = os.path.dirname(wavpath)
    os.makedirs(wavdir, exist_ok=True)
    torchaudio.save(wavpath, audio, sr)