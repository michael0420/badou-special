import numpy as np
import math
import cv2 as cv


class CannyTool:

    def __init__(self, img_path):
        self.image = cv.imread(img_path)
        self.shape = self.image.shape
        self.gray_image = None
        self.gaussian_image = None
        self.M = None  # M: gradient magnitude
        self.NMS = None
        self.out_image = None

    def rgb_to_gray(self):
        self.gray_image = self.image[:, :, 0] * 0.11 + self.image[:, :, 1] * 0.59 + self.image[:, :, 2] * 0.3

    def gaussian_smooth(self, k_size=5, sigma=1.4):
        """
        高斯平滑 降噪
        常用尺寸为 5x5，σ=1.4 的高斯滤波器
        :return:
        """
        # 生成高斯 kernel
        kernel = np.zeros(shape=(k_size, k_size))
        kernel_sum = 0
        for i in range(k_size):
            for j in range(k_size):
                kernel[i, j] = (1 / (2 * math.pi * sigma ** 2)) * np.exp(
                    (-1 / (2 * sigma ** 2) * (np.square(i - 3) + np.square(j - 3))))
                kernel_sum += kernel[i, j]
        # 归一化
        kernel = kernel / kernel_sum

        # 高斯模糊
        self.gaussian_image = np.zeros(self.gray_image.shape)
        H, W = self.gaussian_image.shape

        pad = k_size // 2
        # 保持与原图 尺寸一致，需要对原图像进行填充
        # 输出存储为 (n-k_size +2*pad)/step +1
        pad_img = np.pad(self.gray_image, ((pad, pad), (pad, pad)))  # 默认为constant
        for i in range(H):
            for j in range(W):
                self.gaussian_image[i, j] = np.sum(pad_img[i:i + 5, j:j + 5] * kernel)

    def gradient_magnitude(self):
        """
        计算梯度与幅值
        使用 sobel 算子：
          -1 0 1
          -2 0 2
          -1 0 1

          -1 -2 -1
          0  0  0
          1  2  1
        :return:
        """
        H, W = self.gaussian_image.shape
        dx = np.zeros(shape=[H, W])
        dy = np.zeros(shape=[H, W])
        self.M = np.zeros(shape=[H, W])
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        pad_img = np.pad(self.gaussian_image, ((1, 1), (1, 1)))  # (n-3 +2*1)/1 +1 = n
        for i in range(H):
            for j in range(W):
                dx[i, j] = np.sum(pad_img[i:i + 3, j:j + 3] * kernel_x)
                dy[i, j] = np.sum(pad_img[i:i + 3, j:j + 3] * kernel_y)
                self.M[i, j] = np.sqrt(np.square(dx[i, j]) + np.square(dy[i, j]))
        return dx, dy

    def non_maximum_suppression(self, dx, dy):
        """
        非极大值抑制
        :param dx : x 轴 方向梯度
        :param dy : y 轴 方向梯度
        :return:
        """
        H, W = self.gaussian_image.shape
        self.NMS = np.zeros(shape=[H, W])
        for i in range(1, H - 1):
            for j in range(1, W - 1):

                if self.M[i, j] == 0:
                    continue

                if np.abs(dy[i][j]) > np.abs(dx[i][j]):
                    # 偏向 y轴
                    w = np.abs(dx[i][j]) / np.abs(dy[i][j])
                    g2 = self.M[i, j - 1]
                    g3 = self.M[i, j + 1]
                    # 同方向
                    # g1 g2
                    #    c
                    #    g3 g4
                    if dy[i, j] * dx[i, j] > 0:
                        g1 = self.M[i - 1, j - 1]
                        g4 = self.M[i + 1, j + 1]
                    # 不同方向
                    #    g2 g1
                    #    c
                    # g4 g3
                    else:
                        g1 = self.M[i + 1, j - 1]
                        g4 = self.M[i - 1, j + 1]
                else:
                    # 偏向 x 轴
                    w = np.abs(dy[i][j]) / np.abs(dx[i][j])
                    g2 = self.M[i, j - 1]
                    g3 = self.M[i, j + 1]
                    # 同方向
                    # g1
                    # g2  c  g3
                    #        g4
                    if dy[i, j] * dx[i, j] > 0:
                        g1 = self.M[i - 1, j - 1]
                        g4 = self.M[i + 1, j + 1]
                    # 不同方向
                    #      g4
                    # g2 c g3
                    # g1
                    else:
                        g1 = self.M[i - 1, j + 1]
                        g4 = self.M[i + 1, j - 1]

                temp1 = g1 * w + (1 - w) * g2
                temp2 = g3 * w + (1 - w) * g4
                if self.M[i, j] >= temp1 and self.M[i, j] >= temp2:
                    self.NMS[i, j] = self.M[i, j]
                else:
                    self.NMS[i, j] = 0

    def binary_and_edge_connection(self):
        """
        二值化 和 边连接
        :return:
        """
        # 高阈值是低阈值的三倍
        TL = 0.5 * np.mean(self.NMS)
        TH = 3 * TL
        H, W = self.NMS.shape
        self.out_image = np.zeros(shape=[H, W], dtype=np.uint8)
        stack = []
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                if self.NMS[i, j] >= TH:
                    self.out_image[i, j] = 255
                    stack.append([i, j])
        while stack:
            x, y = stack.pop()
            for i in range(x - 1, x + 2):
                for j in range(y - 1, y + 2):
                    if self.out_image[i, j] >= TH:
                        continue
                    elif self.NMS[i, j] > TL:
                        self.out_image[i, j] = 255
                        stack.append([i, j])

    def save_img(self):
        cv.imwrite(f"canny_detail_gray.png", self.gray_image)
        cv.imwrite(f"k_means_sk_gaussian.png", self.gaussian_image)
        cv.imwrite(f"k_means_sk_gradient_magnitude.png", self.M)
        cv.imwrite(f"k_means_sk_NMS.png", self.NMS)
        cv.imwrite(f"canny_detail_out.png", self.out_image)

    def canny_detail(self):
        """
        canny 边缘检测算法
        :return:
        """
        # 图像灰度化
        self.rgb_to_gray()
        self.gaussian_smooth()
        dx, dy = self.gradient_magnitude()
        self.non_maximum_suppression(dx, dy)
        self.binary_and_edge_connection()
        self.save_img()

    def canny_cv2(self, TL, TH):

        out_image = cv.Canny(self.image, TL, TH)
        cv.imwrite(f"canny_cv2.png", out_image)


if __name__ == '__main__':
    canny_tool = CannyTool("test.jpeg")
    canny_tool.canny_detail()
    canny_tool.canny_cv2(30,100)
