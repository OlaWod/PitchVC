import io

import torch
import numpy as np 
from librosa.filters import mel as librosa_mel_fn


def save_tensor(tensor, save_path):
    f = io.BytesIO()
    torch.save(tensor, f, _use_new_zipfile_serialization=True)
    with open(save_path, "wb") as out_f:
        # Copy the BytesIO stream to the output file
        out_f.write(f.getbuffer())


mel = librosa_mel_fn(sr=16000, n_fft=1024, n_mels=80, fmin=0, fmax=8000)
mel = torch.from_numpy(mel).float()
print(mel.shape)
save_tensor(mel, 'mel_basis.pt')