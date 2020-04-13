import numpy as np
import matplotlib.pyplot as plt

x = np.loadtxt('randomGaussian.data')
plt.hist(x, bins=100)
plt.title('Random data Generator')
plt.show()

x, y, v1, v2 = np.loadtxt('polynomialLinearModel.data', delimiter=',', unpack=True)
plt.plot(x, v1, '.', color='red')
plt.plot(x, v2, '.', color='red')
plt.plot(x, y, '.', label='linear model', color='black')
plt.title('Ground truth')
plt.xlim((-2,2))
plt.ylim((-20, 20))
plt.legend()
plt.show()


xp, yp = np.loadtxt('polynomialLinearModelPredict.data', delimiter=',', unpack=True)
plt.plot(xp, yp, '.', label='predict', color = 'blue')
plt.plot(x, y, '.', label='linear model', color='black')
plt.plot(x, v1, '.', color='red')
plt.plot(x, v2, '.', color='red')
plt.title('Predict result')
plt.xlim((-2,2))
plt.ylim((-20, 20))
plt.legend()
plt.show()