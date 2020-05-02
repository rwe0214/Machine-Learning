from matplotlib import pyplot as plt
import math
import numpy as np

n = np.loadtxt('output/train.data', delimiter=',', unpack=True, usecols=(0,))
x,y,z = np.loadtxt('output/train.data', delimiter=',', unpack=True, skiprows=1)

n=int(n[0])
c0 = n//2-1
c1 = c0+1

plt.subplot(1,3,1)
plt.xlim(min(x)-0.5, max(x)+0.5)
plt.ylim(min(y)-0.5, max(y)+0.5)
plt.scatter(x[0:c0], y[0:c0], color='red')
plt.scatter(x[c1:n-1], y[c1:n-1], color='blue')
plt.title('Ground truth')

wx, wy, w0 = np.loadtxt('output/predict_gradient.data', delimiter=',', unpack=True)
x,y,z = np.loadtxt('output/predict_gradient.data', delimiter=',', unpack=True, skiprows=1)
plt.subplot(1,3,2)
plt.xlim(min(x)-0.5, max(x)+0.5)
plt.ylim(min(y)-0.5, max(y)+0.5)
for i in range(len(x)):
    if z[i] == 0:
        plt.scatter(x[i], y[i], color='red')
    else:
        plt.scatter(x[i], y[i], color='blue')

xp = np.linspace(min(x)-0.5,max(x)+0.5, 1000)
yp = (wx[0]*xp+w0[0]-0.5)/(-wy[0])
plt.plot(xp,yp)
plt.title('Gradient descent')

wx, wy, w0 = np.loadtxt('output/predict_newton.data', delimiter=',', unpack=True)
x,y,z = np.loadtxt('output/predict_newton.data', delimiter=',', unpack=True, skiprows=1)
plt.subplot(1,3,3)
plt.xlim(min(x)-0.5, max(x)+0.5)
plt.ylim(min(y)-0.5, max(y)+0.5)
for i in range(len(x)):
    if z[i] == 0:
        plt.scatter(x[i], y[i], color='red')
    else:
        plt.scatter(x[i], y[i], color='blue')

xp = np.linspace(min(x)-0.5,max(x)+0.5, 1000)
yp = (wx[0]*xp+w0[0]-0.5)/(-wy[0])
plt.plot(xp,yp)
plt.title('Newton\'s method')
plt.savefig('output/result.png')