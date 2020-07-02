import numpy as np

class LDA():
    def __init__(self, datas, labels, k, is_kernel=False):
        self.datas = datas
        self.n = datas.shape[0]
        self.labels = labels
        self.k = k
        self.is_kernel = is_kernel
    
    def __count_labels(self):
        C = np.zeros((self.n, len(np.unique(self.labels))))
        for idx, j in enumerate(np.unique(self.labels)):
            C[self.labels == j, idx] = 1
        return C

    def __get_Sb_Sw(self, C):
        Mj = np.matmul(self.datas.T, C) / np.sum(C, axis = 0)
        M = np.sum(self.datas.T, axis = 1) / self.datas.shape[0]
        
        B = Mj - M[:,None]
        Sb = np.matmul(B * np.sum(C, axis = 0), B.T)
        
        W = self.datas.T - np.matmul(Mj, C.T)
        Sw = np.zeros(Sb.shape)
        for group in np.unique(self.labels):
            w = W[:, self.labels == group]
            Sw += (np.matmul(w, w.T) / w.shape[1])
        return Sb, Sw

    def __get_sorted_eigen(self, A, k):
        eigenvalues, eigenvectors = np.linalg.eigh(A)
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
        C = self.__count_labels()
        Sb, Sw = self.__get_Sb_Sw(C)
        print('solving inverse-problem')
        #try:
        #    Sw_inv = np.linalg.inv(Sw)
        #except:
        #    Sw_inv = np.linalg.pinv(Sw)
        Sw_inv = np.linalg.pinv(Sw)
        print('solving eigen-problem')
        obj_matrix = np.matmul(Sw_inv, Sb)
        value, vector = self.__get_sorted_eigen(obj_matrix, self.k)
        return np.matmul(self.datas, vector), vector