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
    v1 += float(w[i]) * x**i - float(v)
    v2 += float(w[i]) * x**i + float(v)
plt.plot(x, y, label='mean', color = 'black')
plt.plot(x, v1, label='varience', color = 'red')
plt.plot(x, v2, color = 'red')
plt.title('Ground truth')
plt.legend()
plt.savefig('output/Ground truth.png')
plt.show()

x, yp = np.loadtxt('output/bayesianLinearRegressionPredict.data', delimiter=',', unpack=True)
plt.plot(x, yp, 'o', label='income data', color = 'royalblue')

with open('output/bayesianLinearRegressionPredictModel.data') as f:
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
plt.plot(x, y, label='predict model', color='black')
plt.plot(x, v1, color='red')
plt.plot(x, v2, color='red')
plt.title('Predict result')
plt.legend()
plt.savefig('output/Predict result.png')
plt.show()

x, yp = np.loadtxt('output/bayesianLinearRegressionPredict10.data', delimiter=',', unpack=True)
plt.plot(x, yp, 'o', label='income data', color = 'royalblue')

with open('output/bayesianLinearRegressionPredictModel10.data') as f:
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
plt.plot(x, y, label='predict model', color='black')
plt.plot(x, v1, color='red')
plt.plot(x, v2, color='red')
plt.title('After 10 incomes')
plt.legend()
plt.savefig('output/After 10 incomes.png')
plt.show()

x, yp = np.loadtxt('output/bayesianLinearRegressionPredict50.data', delimiter=',', unpack=True)
plt.plot(x, yp, 'o', label='income data', color = 'royalblue')

with open('output/bayesianLinearRegressionPredictModel50.data') as f:
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
plt.plot(x, y, label='predict model', color='black')
plt.plot(x, v1, color='red')
plt.plot(x, v2, color='red')
plt.title('After 50 incomes')
plt.legend()
plt.savefig('output/After 50 incomes.png')
plt.show()
