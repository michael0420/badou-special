import cv2
import numpy as np
import random

# 规范化
class AddNoise:
    """
    添加噪声
    """
    def __init__(self):
        img = cv2.imread("lenna.png")
        h, w, channel = img.shape
        self.h = h
        self.w = w
        self.channel = channel
        self.SNR = 0.5
        self.imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(self.imggray)
        # random.choice([0, 255])


    def addGaussNoise(self):
        """
        添加高斯噪声
        :return:
        """
        img1 = self.imggray.copy()
        noiseNum = int(self.h * self.w * self.SNR + 0.5)
        for n in range(noiseNum):
            # gaussNum = random.choice([0, 255])
            gaussNum = random.gauss(5, 1.5)
            randomh = random.choice([y for y in range(self.h)])
            randomw = random.choice([w for w in range(self.w)])
            newColor = img1[randomh, randomw] + gaussNum
            if newColor < 0:
                newColor = 0
            elif newColor > 255:
                newColor = 255
            img1[randomh, randomw] = newColor
        cv2.imshow("gauss img", img1)
        cv2.waitKey()
        cv2.destroyWindow("gauss img")


    def addPepperSaltNoise(self):
        """
        添加椒盐噪声
        :return:
        """
        img2 = self.imggray.copy()
        noiseNum = int(self.h * self.w * self.SNR + 0.5)
        for n in range(noiseNum):
            num = random.choice([0, 255])
            randomh = random.choice([y for y in range(self.h)])
            randomw = random.choice([w for w in range(self.w)])
            img2[randomh, randomw] = num
        cv2.imshow("peppersalt img", img2)
        cv2.waitKey()
        cv2.destroyWindow("peppersalt img")

    def addThirdNoise(self):
        """
        第三方库添加噪声
        :return:
        """
        pass

# PCA
class featurePCA:
    def __init__(self, X, k):
        self.origin = X
        self.k = k


    def PCA(self):
        # 去中心化 对列求均值
        self.X = self.origin - self.origin.mean(0)
        cov = np.dot(self.X.T, self.X) # 维度 * 维度 所以需要转置
        eig_f, eig_v = np.linalg.eig(cov)   # eig_v 特征向量是在列 不在行

        sorted_index = np.argsort(eig_f)[::-1]
        newVector = eig_v[:, sorted_index[:self.k]]
        # 原始矩阵映射特征举证
        return np.dot(self.X , newVector)


if __name__ == '__main__':
    noise = AddNoise()
    noise.addGaussNoise()
    noise.addPepperSaltNoise()
    X = np.array([[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])

    # X = np.array([[3,2], [1, 4]])
    # eig_f, eig_v = np.linalg.eig(X)
    # sorted(eig_f, reverse=True)

    resVerter = featurePCA(X, 2).PCA()
    print(resVerter)







