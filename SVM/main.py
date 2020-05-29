import numpy as np
import pandas as pd
import libsvm.python.svmutil as svmutil
import matplotlib.pyplot as plt
import grid
import os

def gen_libsvm_format_data(x_data, y_data, output, isKernel=False):
    x_df = pd.read_csv(x_data, header=None, dtype=str)
    y_df = pd.read_csv(y_data, header=None, dtype=str)

    for i in range(len(x_df.columns)):
        x_df[i] = str(i+1) + ':' + x_df[i].astype(str)

    if (isKernel):
        k_df = np.arange(1,len(y_df)+1)
        k_df = pd.DataFrame(k_df)
        k_df = '0:' + k_df[0].astype(str)
        x_df = pd.concat([k_df, x_df], axis = 1)

    format_df = pd.concat([y_df, x_df], axis = 1)
    format_df.to_csv(output, sep=' ', index=False, header=None)

def linear_RBF_kernel(u, v, g):
    linear_x = np.matmul(u, v.T)
    rbf_x = np.sum(u**2, axis=1)[:,None] + np.sum(v**2, axis=1)[None,:] - 2*linear_x
    rbf_x = np.abs(rbf_x) * (-g)
    rbf_x = np.exp(rbf_x)
    return linear_x + rbf_x

#creating directory
if not os.path.exists('./best param'):
    os.mkdir('./best param')
if not os.path.exists('./model'):
    os.mkdir('./model')
if not os.path.exists('./result'):
    os.mkdir('./result')

#preparing the precomputed kernel
print('preparing the precomputed kernel...')
x_train = np.genfromtxt('dataset/X_train.csv', delimiter=',')
x_test = np.genfromtxt('dataset/X_test.csv', delimiter=',')
x_train_precomputed = linear_RBF_kernel(x_train, x_train, 0.03125)
x_test_precomputed = linear_RBF_kernel(x_test, x_test, 0.03125)
np.savetxt('dataset/X_train_precomputed.csv', x_train_precomputed, fmt='%f', delimiter=',')
np.savetxt('dataset/X_test_precomputed.csv', x_test_precomputed, fmt='%f', delimiter=',')

#convert to libsvm-format
print('converting to libsvm-format...')
gen_libsvm_format_data('dataset/X_train_precomputed.csv', 'dataset/Y_train.csv', 'train_precomputed.csv', isKernel=True)
gen_libsvm_format_data('dataset/X_test_precomputed.csv', 'dataset/Y_test.csv', 'test_precomputed.csv', isKernel=True)
gen_libsvm_format_data('dataset/X_train.csv', 'dataset/Y_train.csv', 'train.csv')
gen_libsvm_format_data('dataset/X_test.csv', 'dataset/Y_test.csv', 'test.csv')

#open training data
print('opening training data...')
y_train, x_train = svmutil.svm_read_problem('train.csv')
prob = svmutil.svm_problem(y_train, x_train)

y1, x_train_precomputed = svmutil.svm_read_problem('train_precomputed.csv')
prob_precomputed = svmutil.svm_problem(y1, x_train_precomputed, isKernel=True)


# grid search for the best parameters
print('grid searching...')
grid_search = {
        'linear' : '-t 0 -log2c -5,15,2',
        'polynomial' : '-t 1 -log2c -5,15,2 -log2g -5,15,2 -log2r -3,5,2 -d 4',
        'RBF' : '-t 2 -log2c -5,15,2 -log2g -5,15,2',
        }
grid_search_precomputed = {
        'linear+RBF': '-t 4 -log2c -5,15,2'
        }
best_param = []
kernel_tab = []
kernel = []

for kernel_type, opts in grid_search.items():
    rst, tab, col = grid.find_parameters(prob, opts)
    kernel.append(kernel_type)
    kernel_tab.append(pd.DataFrame(tab, columns=col))
    best_param.append(rst)

for kernel_type, opts in grid_search_precomputed.items():
    rst, tab, col = grid.find_parameters(prob_precomputed, opts)
    kernel.append(kernel_type)
    kernel_tab.append(pd.DataFrame(tab, columns=col))
    best_param.append(rst)

for i in range(len(kernel_tab)):
    kernel_tab[i].to_csv('best param/'+kernel[i]+'.csv')
    print(kernel[i] + ':', end=' ')
    print(best_param[i])

'''
linear kernel:      Best c=0.03125, rate=97.2%
polynomial kernel:  Best c=0.125, g=8192, r=32, d=2, rate=98.4%
RBF kernel:         Best c=2048, g=0.03125, rate=98.74000000000001%
linear+RBF:         Best c=0.03125, rate=97.06
'''

# predict
print('training and predicting...')
import time
start_tic_time = 0
def tic():
    global start_tic_time
    start_tic_time = time.time()
    return start_tic_time

def toc():
    global start_tic_time
    elapsed_time = time.time() - start_tic_time
    return elapsed_time

options = {
    'linear' : '-q -t 0 -c 0.03125',
    'polynomial' : '-q -t 1 -c 0.0125 -g 8192 -r 32 -d 2',
    'RBF' : '-q -t 2 -c 2048 -g 0.03125',
}
options_precomputed = {
    'linear+RBF' : '-q -t 4 -c 0.03125',
}

m = {}
p_label = {}
p_acc = {}
p_val = {}
train_time = {}
test_time = {}
train_flag = True

y_test, x_test = svmutil.svm_read_problem('test.csv')
y1, x_test_precomputed = svmutil.svm_read_problem('test_precomputed.csv')

for kernel_type, opts in options.items():
    print('\tkernel type: {0}\n\t'.format(kernel_type), end='')
    tic()
    if (train_flag):
        m[kernel_type] = svmutil.svm_train(prob, opts)
        svmutil.svm_save_model('model/'+kernel_type+'.model', m[kernel_type])
    else:
        m[kernel_type] = svmutil.svm_load_model('model/'+kernel_type+'.model')
    train_time[kernel_type] = toc()
    tic()
    p_label[kernel_type], p_acc[kernel_type], p_val[kernel_type] = \
    svmutil.svm_predict(y_test, x_test, m[kernel_type])
    test_time[kernel_type] = toc()
    print('\tresult(acc, mse, scc): {0}\n'.format(
        p_acc[kernel_type]))

for kernel_type, opts in options_precomputed.items():
    print('\tkernel type: {0}\n\t'.format(kernel_type), end='')
    tic()
    if (train_flag):
        m[kernel_type] = svmutil.svm_train(prob_precomputed, opts)
        svmutil.svm_save_model('model/'+kernel_type+'.model', m[kernel_type])
    else:
        m[kernel_type] = svmutil.svm_load_model('model/'+kernel_type+'.model')
    train_time[kernel_type] = toc()
    tic()
    p_label[kernel_type], p_acc[kernel_type], p_val[kernel_type] = \
    svmutil.svm_predict(y1, x_test_precomputed, m[kernel_type])
    test_time[kernel_type] = toc()
    print('\tresult(acc, mse, scc): {0}\n'.format(
        p_acc[kernel_type]))
    

# compare result
print('visualizing...')
kernel = list(options.keys()) + list(options_precomputed.keys()) 
p1 = plt.bar(kernel, [test_time[i] for i in kernel], label='test_time', alpha=0.4)
p2 = plt.bar(kernel, [train_time[i] for i in kernel], label='train_time', alpha=0.4, bottom=[test_time[kk] for kk in kernel])
plt.ylabel('time (s)')
plt.xlabel('kernel type')
plt.legend((p1[0], p2[0]), ('test_time', 'train_time'))
plt.savefig('result/times.png')
plt.show()

fig, ytick1 = plt.subplots()
ytick2 = ytick1.twinx()

ytick2.bar(kernel,[m[i].get_nr_sv() for i in kernel], label='# SV', alpha=0.4)
ytick2.set_ylabel('number of SV')
ytick2.legend(loc='upper right')

ytick1.plot(kernel,[p_acc[i][0] for i in kernel], 'b', label='ACC')
ytick1.plot(kernel,[100*p_acc[i][1] for i in kernel], 'r', label='MSE*100')
ytick1.plot(kernel, [100 for i in kernel], 'gray', linestyle='--')
ytick1.set_ylabel('Accuracy rate (%)')
ytick1.legend(loc='upper left')
ytick1.set_xlabel('kernel type')
plt.savefig('result/results.png')
plt.show()

