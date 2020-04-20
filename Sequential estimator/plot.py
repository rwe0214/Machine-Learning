import numpy as np
import matplotlib.pyplot as plt

with open('output/polynomialLinearModel.data') as f:
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

v1 = y - float(v)
v2 = y + float(v)

plt.plot(x, y, label='mean', color = 'black')
plt.plot(x, v1, label='varience', color = 'red')
plt.plot(x, v2, color = 'red')
plt.title('Ground truth')
plt.legend()
plt.savefig('output/Ground truth.png')
plt.show()

xi, yi = np.loadtxt('output/BLR_income.data', delimiter=',', unpack=True)
x, y, v = np.loadtxt('output/BLR_PredictModel.data', delimiter=',', unpack=True)
plt.plot(xi, yi, 'o', label='income data', color = 'royalblue')
plt.plot(x, y, label='predict model', color='black')
plt.plot(x, y+v, color='red')
plt.plot(x, y-v, color='red')
plt.title('Predict result')
plt.legend()
plt.savefig('output/Predict result.png')
plt.show()

x, y, v = np.loadtxt('output/BLR_PredictModel_10.data', delimiter=',', unpack=True)
plt.plot(xi[0:10], yi[0:10], 'o', label='income data', color = 'royalblue')
plt.plot(x, y, label='predict model', color='black')
plt.plot(x, y+v, color='red')
plt.plot(x, y-v, color='red')
plt.title('After 10 incomes')
plt.legend()
plt.savefig('output/After 10 incomes.png')
plt.show()

x, y, v = np.loadtxt('output/BLR_PredictModel_50.data', delimiter=',', unpack=True)
plt.plot(xi[0:50], yi[0:50], 'o', label='income data', color = 'royalblue')
plt.plot(x, y, label='predict model', color='black')
plt.plot(x, y+v, color='red')
plt.plot(x, y-v, color='red')
plt.title('After 50 incomes')
plt.legend()
plt.savefig('output/After 50 incomes.png')
plt.show()