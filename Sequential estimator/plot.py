import numpy as np
import matplotlib.pyplot as plt

x = np.loadtxt('randomGaussian.data')

plt.hist(x, bins=100)
plt.show()