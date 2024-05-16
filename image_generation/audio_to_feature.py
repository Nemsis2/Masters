import torch as th
import torchaudio
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.core.spectrum


def lfb(n_bins=140, n_fft=2048, sr=44100):

    weights = np.zeros((n_bins, int(1 + n_fft // 2)))
    # Center freqs of each FFT bin
    fftfreqs = np.arange(1 + n_fft // 2)*sr / n_fft
    lfbfreqs = np.arange(start=0, stop=fftfreqs[-1], step=fftfreqs[-1]/(n_bins+2))

    ramps = np.subtract.outer(lfbfreqs, fftfreqs)

    fdiff = np.diff(lfbfreqs)
    for i in range(n_bins):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    enorm = 2.0 / (lfbfreqs[2 : n_bins + 2] - lfbfreqs[:n_bins])
    weights *= enorm[:, np.newaxis]

    return weights

# plot the audio
audio, samplerate = torchaudio.load('C_410_cough_0_base.wav')
audio = th.squeeze(audio, 0)
times = th.linspace(0, audio.size()[0]/samplerate, steps=audio.size()[0])

plt.figure()
plt.title("Audio input")
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.plot(times,audio)
plt.show()

# plot the LFB energies
S, n_fft = librosa.core.spectrum._spectrogram(y=np.array(audio),n_fft=2048,hop_length=512,power=2,win_length=None,window="hann",center=True,pad_mode="reflect")
fb = lfb(180, n_fft, samplerate)
fb = librosa.power_to_db(np.abs(np.dot(fb, S)))


plt.figure(figsize=(8, 7))
librosa.display.specshow(fb, sr=samplerate, x_axis='time', y_axis='mel', cmap='magma', hop_length=512)
plt.colorbar(label='dB')
plt.title('LFB Energies (dB)', fontdict=dict(size=18))
plt.xlabel('Time', fontdict=dict(size=15))
plt.ylabel('Frequency', fontdict=dict(size=15))
plt.show()



# plot the mel
mels = librosa.feature.melspectrogram(y=audio.numpy(), sr=samplerate, n_mels=180, n_fft=2048, hop_length=512)
mels_db = librosa.power_to_db(np.abs(mels), ref=np.max)

plt.figure(figsize=(8, 7))
librosa.display.specshow(mels_db, sr=samplerate, x_axis='time', y_axis='mel', cmap='magma', hop_length=512)
plt.colorbar(label='dB')
plt.title('Mel-Spectrogram (dB)', fontdict=dict(size=18))
plt.xlabel('Time', fontdict=dict(size=15))
plt.ylabel('Frequency', fontdict=dict(size=15))
plt.show()


#plot the mfcc
mfcc_ = librosa.feature.mfcc(y=np.array(audio), sr=samplerate, n_mfcc=39, n_fft=n_fft, hop_length=512)


plt.figure(figsize=(8, 7))
librosa.display.specshow(mfcc_, sr=samplerate, x_axis='time', y_axis='mel', cmap='magma', hop_length=512)
plt.colorbar(label='dB')
plt.title('MFCC (dB)', fontdict=dict(size=18))
plt.xlabel('Time', fontdict=dict(size=15))
plt.ylabel('Frequency', fontdict=dict(size=15))
plt.show()