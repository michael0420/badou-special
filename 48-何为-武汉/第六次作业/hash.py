import cv2 as cv
import numpy as np


class HashTools:
    def __init__(self, img_A, img_B):
        self.img_A = img_A
        self.img_B = img_B

    def average_hash(self, img_path):
        """
        均值hash 算法
        :param img_path: 计算hash的 图像路径
        :return:
        """
        # 读取图片
        image = cv.imread(img_path)
        # 缩小尺寸
        """
        INTER_NEAREST – a nearest-neighbor interpolation.
        INTER_LINEAR – a bilinear interpolation (used by default)
        INTER_AREA – resampling using pixel area relation. ...
        INTER_CUBIC – a bicubic interpolation over 4×4 pixel neighborhood.
        INTER_LANCZOS4 – a Lanczos interpolation over 8×8 pixel neighborhood.
        """
        img_8_8 = cv.resize(image, (8, 8), interpolation=cv.INTER_CUBIC)
        # 转灰度值
        img_gray = cv.cvtColor(img_8_8, cv.COLOR_BGR2GRAY)
        # 计算平均值
        avg = np.mean(img_gray)
        # 获取二进制哈希值
        img_hash = ""
        for i in range(8):
            for j in range(8):
                if img_gray[i][j] > avg:
                    img_hash += "1"
                else:
                    img_hash += "0"
        return img_hash

    def diff_hash(self, img_path):
        """
        差值hash
            1. 缩放：图片缩放为8*9，保留结构，除去细节。
            2. 灰度化：转换为灰度图。
            4. 比较：像素值大于后一个像素值记作1，相反记作0。本行不与下一行对比，每行9个像素，
            八个差值，有8行，总共64位
            5. 生成hash：将上述步骤生成的1和0按顺序组合起来既是图片的指纹（hash）。
        """
        image = cv.imread(img_path)
        # 缩小尺寸
        img_8_9 = cv.resize(image, (9, 8), interpolation=cv.INTER_CUBIC)
        # 转灰度值
        img_gray = cv.cvtColor(img_8_9, cv.COLOR_BGR2GRAY)

        # 比较插值 获取二进制哈希值
        img_hash = ""
        for i in range(8):
            for j in range(8):
                if img_gray[i][j] > img_gray[i][j + 1]:
                    img_hash += "1"
                else:
                    img_hash += "0"
        return img_hash

    def discrete_cosine_transform(self, image, N):
        """
        离散余弦变换
        变换 矩阵    F = A f A
        A(i,j)=c(i)cos((j+0.5)*pi*i/N)

        c(i)=sqrt(1/N)  i = 0
        c(i)=sqrt(2/N)  i != 0
        :return:
        """
        A = np.zeros(shape=(N, N), dtype=np.float64)
        for i in range(N):
            for j in range(N):
                c = np.sqrt(1 / N) if i == 0 else np.sqrt(2 / N)
                A[i, j] = c * np.cos((j + 0.5) * np.pi * i / N)
        B = np.dot(A, image)
        res = np.dot(B, A.T)
        return res

    def perception_hash(self, img_path):
        """
            感知哈希（pHash）：
            图片缩放，一般32*32；
            图片灰度化；
            对图片进行离散余弦变换（DCT），转换频域；
            取频域左上角8*8大小（图片的能量都集中在低频部分，低频位于左上角）；
            求平均值，并根据平均值将每一个像素二值化（大于均值为1小于均值为0），生成哈希值；
        :return:
        """
        image = cv.imread(img_path)
        # 缩小尺寸
        img_32_32 = cv.resize(image, (32, 32), interpolation=cv.INTER_AREA)
        # 转灰度值
        img_gray = cv.cvtColor(img_32_32, cv.COLOR_BGR2GRAY)
        img_gray = np.array(img_gray, dtype=np.float64)

        res = self.discrete_cosine_transform(img_gray, 32)

        avg = np.mean(res[0:8, 0:8])
        # 获取二进制哈希值
        img_hash = ""
        for i in range(8):
            for j in range(8):
                if res[i][j] > avg:
                    img_hash += "1"
                else:
                    img_hash += "0"
        return img_hash

    def compare_hash(self, hash_a, hash_b):
        """
        比较 两个 hash 值差异 采用汉明距离
        :param hash_a:
        :param hash_b:
        :return:
        """
        n = 0
        if len(hash_a) != len(hash_b):
            return -1

        for i in range(len(hash_a)):
            if hash_a[i] != hash_b[i]:
                n += 1
        return n

    def compare_img_hash(self):
        """
        比较 三种方法
        :return:
        """
        # 比较 均值哈希
        hash_a = self.average_hash(self.img_A)
        hash_b = self.average_hash(self.img_B)
        n = self.compare_hash(hash_a, hash_b)
        print("----------均值 哈希-----------")
        print("hash_a: {}".format(hash_a))
        print("hash_b: {}".format(hash_b))
        print("差异: {}".format(n))

        # 比较 插值哈希
        hash_a = self.diff_hash(self.img_A)
        hash_b = self.diff_hash(self.img_B)
        n = self.compare_hash(hash_a, hash_b)
        print("----------插值 哈希-----------")
        print("hash_a: {}".format(hash_a))
        print("hash_b: {}".format(hash_b))
        print("差异: {}".format(n))

        # 比较 感知哈希
        hash_a = self.perception_hash(self.img_A)
        hash_b = self.perception_hash(self.img_B)
        n = self.compare_hash(hash_a, hash_b)
        print("----------感知 哈希-----------")
        print("hash_a: {}".format(hash_a))
        print("hash_b: {}".format(hash_b))
        print("差异: {}".format(n))


if __name__ == '__main__':
    hash_tool = HashTools("lenna.png", "lenna_noise.png")
    hash_tool.compare_img_hash()
    pass
