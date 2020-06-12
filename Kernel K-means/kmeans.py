import numpy as np
import sys

class kmeans():
    def __init__(self, data, \
            k=2, \
            method='default', \
            max_iter=100, \
            is_kernel=False, \
            keep_log=False, \
            converge = 5):
        self.data = data
        self.k = k
        self.init = method
        self.max_iter=max_iter
        self.is_kernel = is_kernel
        self.keep_log = keep_log
        self.init = self.__get_init(method)
        self.method = method
        self.converge = converge

    def __get_init(self, method):
        if method == 'kmeans++':
            return self.__kmeanspp
        elif method == 'default':
            return self.__traditional
        else:
            print('ERROR: \'{}\' is not a pre-defined initialize method'.format(method))
            return exit(0)

    def __kmeanspp(self):
        n, d = self.data.shape
        centers = np.array([self.data[np.random.randint(n), :d]])
        for i in range(self.k-1):
            dist = self.euclidean(self.data, centers)
            dist = np.min(dist, axis=1)
            next_center = np.argmax(dist, axis=0)
            centers = np.vstack((centers, self.data[next_center, :]))
        return centers
    
    def __traditional(self):
        return np.array(self.data[np.random.choice(self.data.shape[0], size=self.k, replace=False), :])

    def euclidean(self, u, v):
        return np.matmul(u**2, np.ones((u.shape[1],v.shape[0]))) \
        -2*np.matmul(u, v.T) \
        +np.matmul(np.ones((u.shape[0], v.shape[1])), (v.T)**2)

    def __kernel_trick(self, gram, ck):
        c_count = np.sum(ck, axis=0)
        dist = -2*np.matmul(gram, ck)/c_count + \
            np.matmul(np.ones(ck.shape), (np.matmul(ck.T, np.matmul(gram, ck)))*np.eye(ck.shape[1]))/(c_count**2)
        return dist

    def run(self):
        #initial cluster
        centers = self.init()
        dist = self.euclidean(self.data, centers)
        ck = np.zeros((self.data.shape[0], self.k))
        
        record = []
        ck[np.arange(dist.shape[0]), np.random.randint(self.k, size=dist.shape[0])] = 1
        record.append(ck)
        ck[np.arange(dist.shape[0]), np.argmin(dist, axis=1)] = 1
        record.append(ck)
        record_iter = 0
        for i in range(self.max_iter):
            print('running kernel k-means (k = {0}, {1}).........[{2}/{3}]'.format(self.k, self.method, i, self.max_iter), end='\r')
            #E-step
            if self.is_kernel:
                dist = self.__kernel_trick(self.data, ck)
            else:
                dist = self.euclidean(self.data, centers)

            #M-step
            update_ck = np.zeros(dist.shape)
            update_ck[np.arange(dist.shape[0]),np.argmin(dist, axis=1)] = 1
            delta_ck = np.count_nonzero(np.abs(update_ck - ck))
            update_centers = np.matmul(update_ck.T, self.data)/np.sum(update_ck, axis=0, keepdims=True).T

            record.append(update_ck)
            if delta_ck == 0:
                self.converge -= 1
                if self.converge == 0:
                    record_iter = i+1            
                    if self.keep_log == False:
                        break

            ck = update_ck
            centers = update_centers
        print('running kernel k-means (k = {}, {}).........[\033[92mcomplete at [{}] iterations\033[0m]'.format(self.k, self.method, record_iter))
        return record, record_iter
