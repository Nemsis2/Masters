import numpy as np
import librosa
import matplotlib.pyplot as plt

x = np.arange(-2, 5.25, 0.25)
y = []
for j in range(len(x)):
    if x[j] < 0:
        y.append(0)
    else:
        y.append(x[j])

plt.title("ReLU Activation Function")
plt.plot(x,y)
plt.show()