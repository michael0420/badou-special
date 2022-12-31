import numpy as np

class PCA:
    def __init__(self, X, n):   #矩阵、降到n阶
        self.X = X  #目标矩阵
        self.n = n  #降为n阶

        self.centre_X = self._centre()
        self.cov_matrix = self._cov()
        self.ex_matrix = self._ex_matrix()

    def _centre(self):
        mean = np.array([np.mean(line) for line in self.X.T]) #计算每一个维度的均值
        centre_X = self.X - mean
        print('样本矩阵X的中心化:\n', centre_X)
        return centre_X  #中心化后矩阵

    def _cov(self):
        cov_matrix = np.dot(self.centre_X.T, self.centre_X)/np.shape(self.centre_X)[0]  #计算协方差矩阵
        print('样本矩阵X的协方差矩阵:\n', cov_matrix)
        return cov_matrix

    def _ex_matrix(self):
        index = []
        eigenvalue, eigenvector = np.linalg.eig(self.cov_matrix)
        eigenvalue1 = sorted(eigenvalue, reverse=True)  #特征值从大到小
        for i in list(eigenvalue1):
            index.append(list(eigenvalue).index(i))  #特征值对应原列表位置索引

        eigenvectorT = [np.array(eigenvector)[:, index[i]] for i in range(self.n)] #选取前n个特征，得到对应特征矩阵
        eigenvector1 = np.transpose(eigenvectorT)
        print('%d阶降维转换矩阵:\n' % self.n, eigenvector1)
        return eigenvector1

    def final_matrix(self):
        z = np.dot(self.X, self.ex_matrix)
        print('样本矩阵X的降维矩阵:')
        return z


if __name__ == '__main__':
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35],
                  [42, 45, 11],
                  [9, 48, 5],
                  [11, 21, 14],
                  [8, 5, 15],
                  [11, 12, 21],
                  [21, 20, 25]])
    K = np.shape(X)[1] - 1
    print(PCA(X, K).final_matrix())
