import mnist_loader as mnist
import emlib as em
import random
import numpy as np
from matplotlib import pyplot as plt
import math

train_dataset = mnist.load_MNIST('../dataset/mnist/train-images-idx3-ubyte', '../dataset/mnist/train-labels-idx1-ubyte', bins=2)

#check if mnist_loader works
k = random.randint(0, train_dataset.images_infos[0])
k = 0
print('no.' + str(k) + ':')
for id in range(k, k+10):
    for i in range(0,784):
        if i%28 == 0:
            print('')
        print(train_dataset.bins[id][i], end="")
    print('')
print(train_dataset.images_infos)

# initial parameter
digit_number = 10
# probability of group
k = np.ones(digit_number) / digit_number
# probability of pixel in each group
p = np.random.rand(digit_number, 784)


em.showPofDigit(p)
plt.title("initial probability of {} digits".format(digit_number))
plt.show(block=False)
plt.pause(1)
plt.close()

convergence = 10**-4
max_iter = 100
current_iter = 1
while(True):
    # E step
    # w : 60000 x digit_number
    w = em.Estep(train_dataset.bins, k, p)
    
    # M step
    k_update, p_update = em.Mstep(train_dataset.bins, w, k, p)
    
    em.showPofDigit(p_update)
    plt.title('No. {} iteration'.format(current_iter))
    plt.show(block=False)
    plt.pause(1)
    plt.close()

    k_delta = (k - k_update)
    p_delta = (p - p_update)
    
    k = k_update
    p = p_update

    current_iter += 1
    print('{:d}/{:d} iteration\r'.format(current_iter, max_iter), end='')
    
    if (current_iter > max_iter):
        print('exceed the max iteration')
        break
    
    if(np.count_nonzero(k < (1/6000)) != 0):
        print('not good k, restart EM alorithm')
        k = np.ones(digit_number) / digit_number
        p = np.random.rand(digit_number, 784)
        current_iter = 0
    
    if(np.alltrue(k_delta < convergence) and np.alltrue(p_delta < convergence)):
        print('delta convergence')
        break

test_result = np.zeros((10,10))
test_w = em.Estep(train_dataset.bins, k, p)

test_w_max = np.argmax(test_w, axis=1)

for i in range(test_result.shape[0]):
    for j in range(test_result.shape[1]):
        test_result[i][j] = np.count_nonzero(train_dataset.labels[test_w_max == i] == j)
        print('{:6.0f}'.format(test_result[i][j]), end=' ')
    print('')
        

print(np.argmax(test_result, axis=1), np.argmax(test_result, axis=0))
print(np.argmax(test_result, axis=1)[np.argmax(test_result, axis=0)] == range(10))

#value_index
digit_label = np.argmax(test_result, axis=0)

# show confustion matrix and sensitivity, specificity
for label,digit in enumerate(digit_label):
    TP = np.count_nonzero( train_dataset.labels[test_w_max == digit] == label )
    FP = np.count_nonzero( train_dataset.labels[test_w_max == digit] != label )
    FN = np.count_nonzero( train_dataset.labels[test_w_max != digit] == label )
    TN = np.count_nonzero( train_dataset.labels[test_w_max != digit] != label )
    
    print('')
    print('Confusion Matrix {}:'.format(label))
    print('\t\tPredict number {} Predict not number {}'.format(label, label))
    print('Is number {}\t\t{}\t\t{}'.format(label, TP, FN))
    print('Isn\'t number {}\t\t{}\t\t{}'.format(label, FP, TN))
    print('')
    print('Sensitivity (Successfully predict number {}    : {:.5f}'.format(label, (TP/(TP+FN))))
    print('Specificity (Successfully predict not number {}: {:.5f}'.format(label, (TN/(FP+TN))))
    print('')
    print('-----------------------------------------------------------------------------')

for label in range(len(digit_label)):
    print('{}:'.format(label), end='')
    for i in range(0,784):
        if i%28 == 0:
            print('')
        if p[digit_label[label]][i]> 0.5:
            print('1', end="")
        else:
            print('0', end="")
    print('\n')
em.showPofDigit(p)
plt.title('After {} iteration'.format(current_iter-1))
plt.show()
