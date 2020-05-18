import numpy as np
import random as rd
from scipy.optimize import minimize
from matplotlib import pyplot as plt
np.seterr(divide='ignore',invalid='ignore')

def rq_kernel(xn, xm, length_scale, scale_mixture, amplitude):
	delta = abs(xn-xm)
	return amplitude * (1 + delta**2/(2*scale_mixture*(length_scale**2)))**(-scale_mixture)

def get_K(X, length_scale, scale_mixture, amplitude, beta):
	n = len(X)
	K = np.zeros((n, n), dtype=np.float32)
	for i in range(0, n):
		for j in range(0, n):
			K[i, j] = rq_kernel(X[i], X[j], length_scale, scale_mixture, amplitude)
			if i==j:
				K[i, j] += 1/beta
	return K
			
def get_k_test(test, train, length_scale, scale_mixture, amplitude):
	n = len(train)
	K = np.zeros((n, 1), dtype=np.float32)
	for i in range(0, n):
		K[i, 0] = rq_kernel(train[i], test, length_scale, scale_mixture, amplitude)
	return K

train_x, train_y = np.loadtxt('input.data', delimiter=' ', unpack=True)

#given assumption
beta = 5.0

#initial kernel parameter
length_scale = rd.uniform(1, 10.0)
scale_mixture = rd.uniform(1, 10.0)
amplitude = rd.uniform(1, 10.0)
K = get_K(train_x, length_scale, scale_mixture, amplitude, beta)

test_x = np.arange(-60, 60, 0.5)

test_y = np.zeros((len(test_x), 1))
test_var = np.zeros((len(test_x), 1))
for i in range(len(test_x)):
	k = get_k_test(test_x[i], train_x, length_scale, scale_mixture, amplitude)
	test_y[i] = np.matmul(np.matmul(np.transpose(k), np.linalg.inv(K)), train_y)
	k_new = rq_kernel(test_x[i], test_x[i], length_scale, scale_mixture, amplitude) + 1/beta
	test_var[i] = k_new - np.matmul(np.matmul(np.transpose(k), np.linalg.inv(K)), k)

plt.figure(figsize=(10, 7))
plt.subplot(211)
plt.plot(train_x, train_y, 'o', markeredgecolor='navy', fillstyle='full', markerfacecolor='navy')
s = '  length scale= {:.2f}\nscale mixture= {:.2f}\n      amplitude= {:.2f}'.format(length_scale, scale_mixture, amplitude)
plt.plot(test_x, test_y, color='green', label=s)

y1 = np.array(test_y+1.96*(test_var**0.5)).reshape((240,))
y2 = np.array(test_y-1.96*(test_var**0.5)).reshape((240,))
plt.fill_between(test_x, y1, y2, facecolor='pink', alpha=0.8)
plt.title('random kernel parameters')
plt.legend(loc=1)

#optimize the kernel parameters
def fun(x, args):
	X, Y, beta = args
	K = get_K(X, x[0], x[1], x[2], beta)
	v = np.log(np.linalg.det(K))+np.matmul(np.matmul(np.transpose(train_y), np.linalg.inv(K)), train_y)
	return v

args = [train_x, train_y, beta]
cons = ({'type': 'ineq', 'fun': lambda x: x[0] - 0.1},
		{'type': 'ineq', 'fun': lambda x: x[1] - 0.1},
		{'type': 'ineq', 'fun': lambda x: x[2] - 0.1})

x0 = np.array((length_scale, scale_mixture, amplitude))
res = minimize(fun, x0, args=[train_x, train_y, beta], method='SLSQP', constraints=cons)
print('Value: {}'.format(res.fun))
print('success: {}'.format(res.success))
print('Msg: {}'.format(res.message))
print('Number of iterations: {}'.format(res.nit))
print('Solution: {}'.format(res.x))

length_scale, scale_mixture, amplitude = res.x

K = get_K(train_x, length_scale, scale_mixture, amplitude, beta)
test_x = np.arange(-60, 60, 0.5)

test_y = np.zeros((len(test_x), 1))
test_var = np.zeros((len(test_x), 1))
for i in range(len(test_x)):
	k = get_k_test(test_x[i], train_x, length_scale, scale_mixture, amplitude)
	test_y[i] = np.matmul(np.matmul(np.transpose(k), np.linalg.inv(K)), train_y)
	k_new = rq_kernel(test_x[i], test_x[i], length_scale, scale_mixture, amplitude) + 1/beta
	test_var[i] = k_new - np.matmul(np.matmul(np.transpose(k), np.linalg.inv(K)), k)

plt.subplot(212)
plt.plot(train_x, train_y, 'o', markeredgecolor='navy', fillstyle='full', markerfacecolor='navy')
s = '  length scale= {:.2f}\nscale mixture= {:.2f}\n      amplitude= {:.2f}'.format(length_scale, scale_mixture, amplitude)
plt.plot(test_x, test_y, color='green', label=s)
y1 = np.array(test_y+1.96*(test_var**0.5)).reshape((240,))
y2 = np.array(test_y-1.96*(test_var**0.5)).reshape((240,))
plt.fill_between(test_x, y1, y2, facecolor='pink', alpha=0.8)
plt.title('After optimization')
plt.legend(loc=1)

plt.show()