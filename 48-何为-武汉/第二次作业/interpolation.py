# coding=utf-8

import cv2 as cv
import numpy as np
import math


class Interpolation:

    def __init__(self, img_path):
        self.image = cv.imread(img_path)
        self.shape = self.image.shape
        pass

    def nearest_neighbour_interpolation(self, n):
        """
        :param n 映射倍数
        最近邻插值
        Returns:
        """
        print(self.shape)
        dst_h = int(self.shape[0] * n)
        dst_w = int(self.shape[1] * n)
        dst_img = np.zeros(shape=(dst_h, dst_w, self.shape[2]), dtype=self.image.dtype)

        for x in range(dst_w):
            for y in range(dst_h):
                src_x = int(round(x * 1 / n, 1))
                src_y = int(round(y * 1 / n, 1))
                dst_img[y][x] = self.image[src_y][src_x]

        cv.imwrite(f"nearest_neighbour_interpolation_{n}.png", dst_img)

    def bi_linear_interpolation_left(self, n):
        """
        双线性插值 左上角对其
        :param n 映射倍数
        Returns:
        """
        print(self.shape)
        dst_h = int(self.shape[0] * n)
        dst_w = int(self.shape[1] * n)
        dst_img = np.zeros(shape=(dst_h, dst_w, self.shape[2]), dtype=self.image.dtype)

        for x in range(dst_h):
            for y in range(dst_w):
                src_x = x * 1 / n
                src_y = y * 1 / n
                # 确定原图 四个点 (x0,y0) (x0,y1) (x0,y1) (x1,y1)
                # x1-x0=1  y1-y0=1,注意边界处理
                src_x0 = math.floor(src_x)
                src_y0 = math.floor(src_y)
                src_x1 = min(src_x0 + 1, self.shape[0] - 1)
                src_y1 = min(src_y0 + 1, self.shape[1] - 1)

                # 带入公式计算
                # f(x,y)= (y1 - y)((x1 - x)f(x0, y0) +(x-x0)f(x1,y0)) + (y - y0)((x1 - x)f(x0, y1) +(x-x0)f(x1,y1))
                # 隐式转换 计算出来的是 在赋值时 被隐式转换为int
                dst_img[x][y] = (src_y1 - src_y) * ((src_x1 - src_x) * self.image[src_x0][src_y0] +
                                                    (src_x - src_x0) * self.image[src_x1][src_y0]) + \
                                (src_y - src_y0) * ((src_x1 - src_x) * self.image[src_x0][src_y1] +
                                                    (src_x - src_x0) * self.image[src_x1][src_y1])

        cv.imwrite(f"bi_linear_interpolation_left_{n}.png", dst_img)

    def bi_linear_interpolation_center(self, n):
        """
        双线性插值 左上角对其
        :param n 映射倍数
        Returns:
        """
        print(self.shape)
        dst_h = int(self.shape[0] * n)
        dst_w = int(self.shape[1] * n)
        dst_img = np.zeros(shape=(dst_h, dst_w, self.shape[2]), dtype=self.image.dtype)

        for x in range(dst_h):
            for y in range(dst_w):
                src_x = (x + 0.5) * 1 / n - 0.5
                src_y = (y + 0.5) * 1 / n - 0.5
                # 确定原图 四个点 (x0,y0) (x0,y1) (x0,y1) (x1,y1)
                # x1-x0=1  y1-y0=1,注意边界处理
                src_x0 = math.floor(src_x)
                src_y0 = math.floor(src_y)
                src_x1 = min(src_x0 + 1, self.shape[0] - 1)
                src_y1 = min(src_y0 + 1, self.shape[1] - 1)

                # 带入公式计算
                # f(x,y)= (y1 - y)((x1 - x)f(x0, y0) +(x-x0)f(x1,y0)) + (y - y0)((x1 - x)f(x0, y1) +(x-x0)f(x1,y1))
                # 隐式转换 计算出来的是 在赋值时 被隐式转换为int
                dst_img[x][y] = (src_y1 - src_y) * ((src_x1 - src_x) * self.image[src_x0][src_y0] +
                                                    (src_x - src_x0) * self.image[src_x1][src_y0]) + \
                                (src_y - src_y0) * ((src_x1 - src_x) * self.image[src_x0][src_y1] +
                                                    (src_x - src_x0) * self.image[src_x1][src_y1])

                temp = (src_y1 - src_y) * ((src_x1 - src_x) * self.image[src_x0][src_y0] +
                                           (src_x - src_x0) * self.image[src_x1][src_y0]) + \
                       (src_y - src_y0) * ((src_x1 - src_x) * self.image[src_x0][src_y1] +
                                           (src_x - src_x0) * self.image[src_x1][src_y1])

        cv.imwrite(f"bi_linear_interpolation_center_{n}.png", dst_img)


if __name__ == '__main__':
    interpolation = Interpolation("test.jpeg")

    # 放大 2倍
    interpolation.nearest_neighbour_interpolation(2)
    interpolation.bi_linear_interpolation_left(2.1)
    interpolation.bi_linear_interpolation_center(2.3)

