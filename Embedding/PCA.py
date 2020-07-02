import numpy as np

class PCA():
    def __init__(self, datas, k, is_kernel=False):
        self.datas = datas
        self.k = k
        self.is_kernel = is_kernel

    def __get_covariance(self, datas):
        n = datas.shape[0]
        if self.is_kernel:
            N1 = np.ones((n,n))/n
            S = (datas - np.matmul(N1, datas) - np.matmul(datas, N1) + np.matmul(np.matmul(N1, datas), N1))
            return S
        else:
            S = datas - np.sum(datas, axis=0)/n
            S = np.matmul(S, S.T)/n
            return S

    def __get_sorted_eigen(self, A, k):
        eigenvalues, eigenvectors = np.linalg.eig(A)
        sorted_idx = np.flip(np.argsort(eigenvalues))
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
        S = self.__get_covariance(self.datas)
        e_vals, e_vecs = self.__get_sorted_eigen(S, self.k)
        mean_datas = self.datas - np.sum(self.datas, axis=0)/self.datas.shape[0]
        if self.is_kernel:
            W = e_vecs
            W /= np.sqrt(e_vals)
            N1 = np.ones(self.datas.shape)/self.datas.shape[0]
            pca_space = np.matmul((self.datas - np.matmul(N1, self.datas)), W)
        else:
            W = np.matmul(mean_datas.T, e_vecs)
            W /= np.sqrt(e_vals)
            pca_space = np.matmul(self.datas, W)
        return pca_space, W
    