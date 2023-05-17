import numpy as np
import librosa
import matplotlib.pyplot as plt

sr = 44100
n=10
mel_basis = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=n,fmin=0, fmax=sr / 2)
f = np.linspace(0, sr/2, int((2048/2)+1))
for i in range(n):
    plt.plot(f,mel_basis[i])

plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.show()