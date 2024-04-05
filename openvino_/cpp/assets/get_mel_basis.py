import numpy as np 
from librosa.filters import mel as librosa_mel_fn


mel = librosa_mel_fn(sr=16000, n_fft=1024, n_mels=80, fmin=0, fmax=8000)
print(mel.shape)
np.save('mel_basis.npy', mel)