from sklearn.decomposition import PCA
import numpy as np

class MyPCA():
    def __init__(self, n_components) -> None:
        self.n_components = n_components
    
    def fit_transform(self, X):
        self.X = X
        self._central() # 中心化
        self._getconv() # 获取协方差矩阵
        self._get_eigenvalue_vectors()  # 获取协方差矩阵的特征值和特征向量
        self._getw()                    # 根据降维后的数据获取特征矩阵
        return np.dot(self.X, self.w)
        
    def _central(self):
        self.X = self.X - self.X.mean(axis=0)
        print('中心化后的数据为：')
        print(self.X)
    
    def _getconv(self):
        self.c = (np.dot(self.X.T, self.X)) / (self.X.shape[0] - 1)
        print('该数据的协方差矩阵为:')
        print(self.c)
    
    def _get_eigenvalue_vectors(self):
        self.values, self.vectors = np.linalg.eig(self.c)
        print('协方差矩阵的特征值为：')
        print(self.values)
        print('协方差矩阵的特征向量为：')
        print(self.vectors)
    
    def _getw(self):
        idx = np.argsort(-self.values)
        self.w = self.vectors[:, idx[:self.n_components]]
        print('特征矩阵为：')
        print(self.w)

def myPca(X, use_module=True):
    if use_module:
        print('使用sklearn库进行PCA降维')
        pca = PCA(n_components=2)
        y = pca.fit_transform(X)
        print('降维后的数据为：')
        print(y)
    else:
        print('使用自主实现类进行PCA降维')
        pca = MyPCA(n_components=2)
        y = pca.fit_transform(X)
        print('降维后的数据为：')
        print(y)

if __name__ == "__main__":
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14],
                  [8,  5,  15],
                  [11, 12, 21],
                  [21, 20, 25]])
    myPca(X, use_module=True)
    myPca(X, use_module=False)