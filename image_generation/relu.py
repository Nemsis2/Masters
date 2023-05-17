import numpy as np
import librosa
import matplotlib.pyplot as plt

x = []
for i in range(-10,10):
    x.append(i)

y = []
for j in range(len(x)):
    if x[j] < 0:
        y.append(0)
    else:
        y.append(x[j])
plt.title("ReLU Activation Function")
plt.plot(x,y)
plt.show()