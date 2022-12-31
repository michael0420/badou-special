import numpy as np


class PCA:
    def __init__(self, x, n_components):
        self.x = x
        self.n_components = n_components
        self.n_features_ = 0
        self.covariance, self.components_ = np.array([]), np.array([])

    def fit_transform(self):
        self.n_features_ = self.x.shape[1]
        # 求协方差矩阵
        self.x = self.x - np.mean(self.x.T, axis=1)
        self.covariance = np.dot(self.x.T, self.x) / (self.x.shape[0] - 1)
        # 求协方差矩阵的特征值和特征向量
        eig_val, eig_vectors = np.linalg.eig(self.covariance)
        # 获得降序排列特征值的序号
        idx = np.argsort(-eig_val)
        # 降维矩阵
        self.components_ = -eig_vectors[:, idx[:self.n_components]]
        # 对X进行降维
        return np.dot(self.x, self.components_)


class PCA2:
    """采用特征值累计值计算主成分分析"""
    def __init__(self, x, threshold=0.9):
        self.threshold = threshold
        self.x = x
        self.center, self.covariance = np.array([]), np.array([])

    def fit_transform(self):
        # 去中心化
        self.center = self.x - self.x.mean(axis=0)
        # 构造协方差矩阵
        self.covariance = np.cov(self.x, rowvar=False, bias=False)
        # 求协方差矩阵的特征值和特征向量
        w, v = np.linalg.eig(self.covariance)
        # 构造符合的特征矩阵
        cond = (w / w.sum()).cumsum() >= self.threshold
        # 构造特征矩阵
        v = -v[:, :cond.argmax() + 1]
        # 将原来矩阵映射到新的空间中
        return self.center.dot(v)


if __name__ == '__main__':
    _x = np.loadtxt('./data/demo.csv', dtype=np.float, delimiter=',')
    pca = PCA(_x, n_components=2)
    print(pca.fit_transform())
    pca2 = PCA2(_x)
    print(pca2.fit_transform())
