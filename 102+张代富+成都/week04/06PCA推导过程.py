import numpy as np


class PCA:
    def __init__(self, x, k):
        """初始化"""
        self.x = x
        self.k = k
        self.center_x = np.array([])
        self.c, self.u, self.z = np.array([]), np.array([]), np.array([])

        self.centralization()
        self.covariance()
        self.transform_matrix()
        self.get_result()

    def centralization(self):
        """矩阵中心化"""
        mean = np.mean(self.x.T, axis=1)
        self.center_x = self.x - mean

    def covariance(self):
        """协方差"""
        ns = self.center_x.shape[0]
        self.c = np.dot(self.center_x.T, self.center_x) / (ns - 1)

    def transform_matrix(self):
        """降维转换矩阵"""
        # 特征值和特征向量
        a, b = np.linalg.eig(self.c)
        index = np.argsort(-1 * a)
        self.u = -b[:, index[:self.k]]

    def get_result(self):
        self.z = np.dot(self.center_x, self.u)


if __name__ == '__main__':
    _x = np.loadtxt('./data/demo.csv', dtype=np.float, delimiter=',')
    pca = PCA(_x, 2)
    print(pca.z)

    # 和sklearn中进行比较
    from sklearn.decomposition import PCA

    p = PCA(2)
    p.fit(_x)
    print(f"{'-' * 15}PCA{'-' * 15}")
    print(p.transform(_x))
