"""
@author: Rai
手动实现canny检测边缘
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math


def my_cany(sigma):
    img = cv2.imread('lenna.png')
    # img_gray = int(img[:, :, 0] * 0.11 + img[:, :, 1] * 0.59 + img[:, :, 2] * 0.3)
    img_gray = np.uint8(img.mean(axis=2))

    # 1.高斯平滑
    ksize = int(np.round(6 * sigma + 1))
    if ksize % 2 == 0:
        ksize += 1
    v = [i - ksize // 2 for i in range(ksize)]
    c1 = 1 / (2 * math.pi * sigma ** 2)
    c2 = -1 / (2 * sigma ** 2)
    gauss_kernel = np.zeros((ksize, ksize))
    for i in range(ksize):
        for j in range(ksize):
            # 高斯函数前的常数可以不用计算，会在归一化的过程中给消去
            # gauss_kernel[i, j] = c1 * math.exp(c2 * (v[j]**2 + v[ksize-i-1]**2))
            gauss_kernel[i, j] = math.exp(c2 * (v[j] ** 2 + v[ksize - i - 1] ** 2))
    gauss_fileter = gauss_kernel / gauss_kernel.sum()
    # print('gauss filter:', gauss_fileter)
    dx, dy, c = img.shape
    img_new = img_gray.copy()
    img_new = np.pad(img_new, ksize // 2)
    img_gauss = np.zeros((dx, dy))
    for i in range(dx):
        for j in range(dy):
            img_gauss[i, j] = np.sum(img_new[i:i + ksize, j:j + ksize] * gauss_fileter, dtype=np.uint8)
    img_gauss = np.clip(img_gauss, 0, 255)
    img_merge = np.hstack([np.uint8(img_gray), img_gauss])
    plt.figure(1)
    plt.imshow(img_merge, cmap='gray')
    plt.title('Left: gray img       Right: gauss img', color='blue')
    plt.axis('off')
    plt.savefig('Gaussian img.png')
    # cv2.imwrite('Gaussian img.png', img_gauss)
    # cv2.imshow('Left: gray img, right: gauss img', img_merge)
    # cv2.waitKey(0)

    # 2.求梯度。用sobel算子矩阵检测图像的水平、垂直、对焦边缘
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # 检测垂直边缘
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # 检测水平边缘
    img_Gx = np.zeros(img_gauss.shape)
    img_Gy = np.zeros(img_gauss.shape)
    img_G = np.zeros(img_gauss.shape)
    img_pad = np.pad(img_gauss, 1)
    for i in range(dx):
        for j in range(dy):
            img_Gx[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)
            img_Gy[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)
            img_G[i, j] = np.sqrt(img_Gx[i, j] ** 2 + img_Gy[i, j] ** 2)
    img_Gx[img_Gx == 0] = 1E-8
    tan_v = img_Gy / img_Gx  # 梯度
    img_Gx = np.where(img_Gx < 0, abs(img_Gx), img_Gx)
    img_Gx = np.clip(img_Gx, 0, 255)
    img_Gy = np.where(img_Gy < 0, abs(img_Gy), img_Gy)
    img_Gy = np.clip(img_Gy, 0, 255)
    img_G_plot = np.uint8(np.where(img_G > 255, 255, img_G))
    img_merge = np.hstack([img_Gx, img_Gy, img_G_plot])
    plt.figure(2)
    plt.imshow(img_merge, cmap='gray')
    plt.title('Horizontal edge        Vertical edge        Digonal edge', color='blue')
    plt.axis('off')
    plt.savefig('Sobel img.png')

    # 3.非极大值抑制
    img_NMS = np.zeros(img_G.shape)
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            tmp = img_G[i - 1:i + 2, j - 1:j + 2]
            flag = True  # 默认8领域内的中心值是最大值
            if tan_v[i, j] <= -1:
                # 双线性插值,由[tmp[0, 0] * (-1/tan_v[i, j) + (1 - (-1/tan_v[i,j]) * tmp[0, 1]]可化简得到下式
                num1 = (tmp[0, 1] - tmp[0, 0]) / tan_v[i, j] + tmp[0, 1]
                # 双线性插值,由[tmp[2, 2] * (-1/tan_v[i, j) + (1 - (-1/tan_v[i,j]) * tmp[2, 1]]可化简得到下式
                num2 = (tmp[2, 1] - tmp[2, 2]) / tan_v[i, j] + tmp[2, 1]
                if img_G[i, j] <= num1 or img_G[i, j] <= num2:
                    flag = False
            elif tan_v[i, j] >= 1:
                num1 = (tmp[0, 2] - tmp[0, 1]) / tan_v[i, j] + tmp[0, 1]
                num2 = (tmp[2, 0] - tmp[2, 1]) / tan_v[i, j] + tmp[2, 1]
                if img_G[i, j] <= num1 or img_G[i, j] <= num2:
                    flag = False
            elif tan_v[i, j] > 0:
                num1 = (tmp[0, 2] - tmp[1, 2]) * tan_v[i, j] + tmp[1, 2]
                num2 = (tmp[2, 0] - tmp[1, 0]) * tan_v[i, j] + tmp[1, 0]
                if img_G[i, j] <= num1 or img_G[i, j] <= num2:
                    flag = False
            elif tan_v[i, j] <= 0:
                num1 = (tmp[1, 2] - tmp[2, 2]) * tan_v[i, j] + tmp[1, 2]
                num2 = (tmp[1, 0] - tmp[0, 0]) * tan_v[i, j] + tmp[1, 0]
                if img_G[i, j] <= num1 or img_G[i, j] <= num2:
                    flag = False
            if flag:
                img_NMS[i, j] = img_G[i, j]
    plt.figure(3)
    img_NMS_plot = np.uint8(np.where(img_NMS > 255, 255, img_NMS))
    img_merge = np.hstack([img_G_plot, img_NMS_plot])
    plt.imshow(img_merge, cmap='gray')
    plt.axis('off')
    plt.title('Left: Sobel image,  Right: NMS image', color='blue')
    cv2.imwrite('NMS img.png', img_NMS_plot)

    # 4.双阈值检测，连接边缘
    edge_lower = img_G.mean() * 0.5
    edge_upper = edge_lower * 3
    img_cany = img_NMS.copy()
    img_cany[img_cany >= edge_upper] = 255  # 强边缘
    img_cany[img_cany <= edge_lower] = 0   # 抑制
    near_field_8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            # 弱边缘
            if img_cany[i, j] > edge_lower and img_cany[i,j] < edge_upper:
                tmp = img_cany[i-1:i+2, j-1:j+2] * near_field_8
                if np.max(tmp) >= edge_upper:
                    img_cany[i, j] = 255
                else:
                    img_cany[i, j] = 0

    img_cany = np.uint8(img_cany)
    img_merge = np.hstack([img_NMS_plot, img_cany])
    plt.figure(4)
    plt.imshow(img_merge, cmap='gray')
    plt.title('Left: NMS image,       Right: two boundary detection', color='blue')
    plt.axis('off')
    cv2.imwrite('Canny img.png', img_cany)
    plt.show()


if __name__ == "__main__":
    my_cany(0.5)
