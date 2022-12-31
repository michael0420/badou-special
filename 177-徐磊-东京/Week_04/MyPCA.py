# -*- coding utf-8 -*-
'''
手动PCA
'''

import numpy as np


class MyPCA(object):
    def __init__(self, X, dim):
        self.X = X
        self.dim = dim
        self.centraX = self.centralization()  # 数据中心化
        self.C = self.convariationCalc()  # 求协方差
        self.R = self.eigenvalueCalc()  # 求特征值和特征向量
        self.X_new = self.transform()  # 求解降维后的数据

    def centralization(self):
        X_mean = [np.mean(v) for v in self.X.T]  # 求解样本X各个特征均值
        print('样本数据特征均值:\n', X_mean)
        centraX = self.X - X_mean
        print('样本数据中心化:\n', centraX)
        return centraX

    def convariationCalc(self):
        C = np.dot(self.centraX.T, self.centraX) / (np.shape(self.centraX)[0] - 1)
        print("样本数据协方差矩阵：\n", C)
        return C

    def eigenvalueCalc(self):
        eig_value, eig_vector = np.linalg.eig(self.C)
        print("样本数据协方差矩阵的特征值：\n", eig_value)
        print("样本数据协方差矩阵的特征向量:\n", eig_vector)
        id_des = np.argsort(eig_value)[::-1]
        eig_vector_tranf = np.array([eig_vector[:,i] for i in id_des[:self.dim]]).T
        print("%d阶降维转换矩阵:\n" % self.dim, eig_vector_tranf)
        return eig_vector_tranf

    def transform(self):
        X_new = np.dot(self.X, self.R)
        print("降维后的数据：\n", X_new)
        return X_new


if __name__ == "__main__":
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

    dim = np.shape(X)[1] - 1
    MyPCA(X, dim)
