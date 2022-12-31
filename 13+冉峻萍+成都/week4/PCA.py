import numpy as np

class CPCA(object):
    def __init__(self, X, K):
        self.X = X          # 样本矩阵 X
        self.K = K          # K 阶降维矩阵的 K 值
        self.centrX = []    # 矩阵 X 的中心化
        self.C = []         # 样本集的协方差矩阵 C
        self.U = []         # 样本矩阵 X 的降维转换矩阵
        self.Z = []         # 样本矩阵 X 的降维矩阵 Z
        
        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()  # Z = XU 求得
        
    # 1. 求矩阵 X 的中心化
    def _centralized(self):
        print("样本矩阵 X：\n", self.X)
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T]) # 样本矩阵的特征均值
        print("样本集的特征均值：\n", mean)
        centrX = self.X - mean # 样本集的中心化
        print("样本矩阵 X 的中心化 centrX：\n", centrX)
        return centrX
        
    # 2. 求样本矩阵 X 的协方差矩阵 C
    # 注意这里的样本矩阵是中心化后的矩阵，不是原始矩阵
    def _cov(self):
        ns = np.shape(self.centrX)[0] # 样本的样例总数
        # 做了中心化的矩阵，协方差计算公式：
        # D = (1/m)(Z^T*Z)
        C = np.dot(self.centrX.T, self.centrX)/(ns - 1)
        print("样本矩阵 X 的协方差矩阵 C：\n", C)
        return C
        
    # 3. 求特征矩阵
    def _U(self):
        # 先求 X 的协方差矩阵 C 的特征值和特征向量
        a,b = np.linalg.eig(self.C)
        print("样本集的协方差矩阵 C 的特征值：\n", a)
        print("样本集的协方差举证 C 的特征向量：\n", b)
        ind = np.argsort(-1*a)
        # 构建 K 阶降维的降维矩阵 U
        UT = [b[:,ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print("%d 阶降维转换矩阵 U：\n" % self.K, U)
        return U
        
    def _Z(self):
        # 按照 Z=XU 求降维矩阵 Z
        Z = np.dot(self.X, self.U)
        print("X shape：", np.shape(self.X))
        print("U shape：", np.shape(self.U))
        print("Z shape：", np.shape(Z))
        print("样本矩阵 X 的降维矩阵 Z：\n", Z)
        return Z

if __name__ == '__main__':
    # 10 个样本 3 特征的样本集，行为样例，列为特征维度
    # 所以说这里有 10 个样例，3 个特征维度
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
                   
    # 因为源数据集只有三个维度，所以我们再做降维处理的时候只能够降到两维
    K = np.shape(X)[1] - 1
    #print("样本集（10行三列，10 个样例，每个样例 3 个特征）：\n", X)
    pca = CPCA(X,K)