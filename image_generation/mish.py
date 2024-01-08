import numpy as np
import librosa
import matplotlib.pyplot as plt

x = np.arange(-10, 10, 0.25)

y = []
for j in range(len(x)):
    y.append(x[j] * np.tanh(np.log(1+np.exp(x[j]))))


plt.title("Mish Activation Function")
plt.plot(x,y)
plt.show()