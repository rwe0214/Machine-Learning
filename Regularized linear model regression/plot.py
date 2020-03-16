import matplotlib.pyplot as plt
import numpy as np

x, y = np.loadtxt('datapoint.data', delimiter=',', unpack=True)
plt.scatter(x,y,marker= ".")
lse = np.loadtxt('output/LSE.txt', delimiter=',')
newton = np.loadtxt('output/newton.txt', delimiter=',')

upper = max(x)
lower = min(x)

x1 = np.arange(lower-2, upper+2)
y1 = 0.0
for i, cof in enumerate(lse):
    y1 += (cof * (x1 ** (len(lse)-i-1)))

plt.plot(x1, y1, 'r', label='rLSE')

x2 = np.arange(lower-2, upper+2)
y2 = 0.0
for i, cof in enumerate(newton):
    y2 += (cof * (x2 ** (len(newton)-i-1)))

plt.plot(x2, y2, 'g', label='newton\'s')

plt.xlabel('x')
plt.ylabel('y')
plt.title('rLSE v.s. Newton\'s Method')
plt.legend()
plt.savefig('output/rLSE v.s. Newton\'s Method.png')
plt.show()