import numpy as np
from scipy.linalg import eig
from kmeans import kmeans

class spectral_clustering():
    def __init__(self, data_similarity , \
            k=2, \
            normalize=False, \
            keep_log=False):
        self.k = k
        self.W = data_similarity
        self.D = np.sum(data_similarity, axis=1, keepdims=True) * np.eye(data_similarity.shape[0])
        self.normalize = normalize
        self.keep_log = keep_log

    def __eig(self, A):
        if self.normalize:
            sqrt_D = np.sqrt(self.D)
            neg_sqrt_D = np.linalg.inv(sqrt_D)
            N = np.matmul(np.matmul(neg_sqrt_D, A), sqrt_D)
            eigenvals, eigenvecs = np.linalg.eig(N)
            eigenvecs = np.matmul(neg_sqrt_D, eigenvecs.real)
            return eigenvals, eigenvecs
        else:
            return np.linalg.eig(A)

    def __get_sorted_k_eigen(self, A, k):
        eigenvalues, eigenvectors = self.__eig(A)

        sorted_idx = np.argsort(eigenvalues)
        sorted_eigenvalues = []
        sorted_eigenvectors = []
        for i in range(k):
            vector = eigenvectors[:, sorted_idx[i]]
            sorted_eigenvectors.append(vector[:, None])
            sorted_eigenvalues.append(eigenvalues[sorted_idx[i]])        
        sorted_eigenvalues = np.array(sorted_eigenvalues)
        sorted_eigenvectors = np.concatenate(sorted_eigenvectors, axis=1)
    
        return sorted_eigenvalues, sorted_eigenvectors       

    def run(self):
        self.L = self.D - self.W
        print('compute eigenvalues and eigenvectors..........', end='\r')
        k_eigenvalues, k_eigenvectors = self.__get_sorted_k_eigen(self.L, self.k)
        print('compute eigenvalues and eigenvectors..........[\033[92mcomplete\033[0m]')
        km = kmeans(k_eigenvectors, k=self.k, keep_log=self.keep_log)
        return km.run(), k_eigenvalues, k_eigenvectors
        