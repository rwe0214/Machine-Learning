import numpy as np
from matplotlib import pyplot as plt
import math

def showPofDigit(m, row=2, col=5, imsize=28):
    m2 = m.reshape((row*col,imsize,imsize))
    m2 = (np.concatenate((m2[:col]), axis=1), np.concatenate((m2[col:]), axis=1))
    m2 = np.concatenate(m2, axis=0)
    plt.figure(figsize=(10,5))
    plt.imshow(m2, cmap='gray', vmin=0, vmax=1)
    plt.xticks([i*imsize + (imsize/2) for i in range(col)],[str(i) for i in range(col)])
    plt.yticks([i*imsize + (imsize/2) for i in range(row)],[str(i*col) for i in range(row)])

# zero threshold
zthreshold = 10**-10
def mylog(m):
    return np.log(np.where(m > zthreshold, m, zthreshold))

# return w responsibility
def Estep(X, k, p):
    pixel1 = np.matmul(X, mylog(p.T))
    pixel0 = np.matmul(1 - X, mylog(1 - p.T))
    w = pixel1 + pixel0 + mylog(k)
    
    #normalize for prevent overflow while calling exp()
    w = (w.T - np.max(w, axis=1)).T

    w = np.exp(w)
    w = (w.T / np.sum(w, axis=1)).T
    return w

# return update value
def Mstep(X, w, k, p):
    w_sum = np.sum(w, axis=0)

    k_update = w_sum / w.shape[0]
    p_update = (np.matmul(X.T, w) / w_sum).T
    
    return k_update, p_update