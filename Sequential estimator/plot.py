import numpy as np
import matplotlib.pyplot as plt

x = np.loadtxt('randomGaussian.data')
plt.hist(x, bins=100)
plt.title('Random data Generator')
plt.show()

x, y, z = np.loadtxt('polynomialLinearModel.data', delimiter=',', unpack=True)
plt.plot(x, y, '.', label='predict')
plt.plot(x, z, '-', label='linear model')
plt.title('Polynomial Linear Model Generator')
plt.legend()
plt.show()