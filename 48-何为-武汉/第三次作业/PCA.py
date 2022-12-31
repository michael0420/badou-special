import numpy as np
from sklearn.decomposition import PCA


class PCATool:
    def __init__(self, data, k):
        """
        :param data: n*m  n 组数据，m 维度
        :param k: 降至多少维 k <= m
        """
        self.data = np.array(data)
        self.shape = np.shape(self.data)
        self.k = k
        pass

    def pca_detail(self):
        # 计算每个维度的均值
        mean = [np.mean(self.data[:, i]) for i in range(self.shape[1])]
        print(f"样本均值：\n{mean}")
        # 每个维度 去均值
        data = self.data - mean
        print(f"去均值样：\n{data}")
        # 计算协方差矩阵
        cov = np.dot(data.T, data) / (self.shape[0] - 1)
        print(f"样本协方差矩阵：\n{cov}")
        # 计算特征值
        a, b = np.linalg.eig(cov)
        print(f"协方差 特征值：\n{a}")
        print(f"协方差 特征向量值：\n{b}")
        # 特征值排序
        sort_index = np.argsort(-a)
        print(sort_index)
        eig_mat = b[:, sort_index[:self.k]]
        print(f"样本 投影矩阵：\n{eig_mat}")
        # 数据投影（降维）
        final_data = np.dot(self.data, eig_mat)
        print(f"样本转换后的数据：\n{final_data}")
        pass

    def pca_sk_learn(self):
        print(f" pca_sk_learn 计算")
        print(f" 样本数据：\n{self.data}")
        # 样本数据 降到 k 维
        pca = PCA(n_components=self.k)
        # # 输入样本数据 训练
        pca.fit(self.data)
        #
        print(f"模型参数：\n{pca.get_params()}")
        print(f"样本协方差矩阵：\n{pca.get_covariance()}")
        print(f"样本映射矩阵：\n{pca.components_}")
        # 样本数据映射
        print(f" 样本数据：\n{self.data}")
        final_data = pca.transform(self.data)
        print(f"样本转换后的数据(样本去中心化了 之后转换)：\n{final_data}")
        print(pca.explained_variance_ratio_)  # 输出贡献率


if __name__ == '__main__':
    # data = [[10, 29, 15],
    #         [15, 13, 46],
    #         [23, 30, 21],
    #         [11, 35, 9],
    #         [42, 11, 45],
    #         [9, 5, 48],
    #         [11, 14, 21],
    #         [8, 15, 5],
    #         [11, 21, 12],
    #         [21, 25, 20]]

    data = [[10, 15, 29],
            [15, 46, 13],
            [23, 21, 30],
            [11, 9, 35],
            [42, 45, 11],
            [9, 48, 5],
            [11, 21, 14],
            [8, 5, 15],
            [11, 12, 21],
            [21, 20, 25]]
    pca = PCATool(data, 2)

    pca.pca_detail()
    pca.pca_sk_learn()
