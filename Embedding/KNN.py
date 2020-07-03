import numpy as np


class KNN():
    def __init__(self, train_datas, test_datas, labels, k=5):
        self.train_datas = train_datas
        self.test_datas = test_datas
        self.labels = labels
        self.k = k

    def __eucidiance(self, U, V):
        return np.matmul(U**2, np.ones(
            (U.shape[1], V.shape[0]))) - 2 * np.matmul(U, V.T) + np.matmul(
                np.ones((U.shape[0], V.shape[1])), (V.T)**2)

    def run(self):
        dist = self.__eucidiance(self.test_datas, self.train_datas)
        closet = np.argsort(dist, axis=1)

        y = []
        for i in range(self.test_datas.shape[0]):
            for j in range(self.k):
                y.append(self.labels[closet[i][j]])
        y = np.array(y)
        y = y.reshape(self.test_datas.shape[0], self.k)

        count = []
        for i in range(len(self.labels)):
            count.append(np.count_nonzero(y == self.labels[i], axis=1))
        count = np.vstack(count)
        y = np.argmax(count, axis=0)
        predict = [None] * self.test_datas.shape[0]
        for i in range(self.test_datas.shape[0]):
            predict[i] = self.labels[y[i]]

        return predict
