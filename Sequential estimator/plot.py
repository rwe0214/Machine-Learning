import numpy as np
import matplotlib.pyplot as plt

x = np.loadtxt('randomGaussian.data')
plt.hist(x, bins=100)
plt.title('Random data Generator')
plt.show()

with open('polynomialLinearModel.data') as f:
    temp = f.read().splitlines()
    w = []
    w = temp[0].split(',')
    v = temp[1]
x = np.arange(-2, 2, 0.1)
y = 0.0
v1 = 0.0
v2 = 0.0
for i in range(len(w)):
    y += float(w[i]) * x**i
    v1 += float(w[i]) * x**i - float(v)
    v2 += float(w[i]) * x**i + float(v)
plt.plot(x, y, label='mean', color = 'black')
plt.plot(x, v1, label='varience', color = 'red')
plt.plot(x, v2, color = 'red')
plt.title('Ground truth')
plt.legend()
plt.show()

xp, yp = np.loadtxt('polynomialLinearModelPredict.data', delimiter=',', unpack=True)
plt.plot(xp, yp, '.', label='predict', color = 'skyblue')
plt.plot(x, y, label='linear model', color='black')
plt.plot(x, v1, color='red')
plt.plot(x, v2, color='red')
plt.title('Predict result')
plt.legend()
plt.show()