import cv2
import numpy as np
import matplotlib.pyplot as plt

# BGR
image = cv2.imread('./lenna.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# 灰度化1
image_gray = image[:, :, 1]

# 灰度化2
image_gray2 = (image[:, :, 2] * 0.11 + image[:, :, 1] * 0.59 + image[:, :, 0] * 0.3).astype(np.uint8)

# 灰度化3
image_gray3 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 灰度化4 二值化
image_gray4 = np.where(image_gray2 >= 255 // 2, 255, 0).astype(np.uint8)

# 灰度化5 0-255 转化为0-1 二值化
image_gray5 = np.where(image_gray2 / 255 >= 0.5, 1, 0).astype(np.float32)

# 灰度化6 为0-1 二值化 黑白反转
image_gray6 = np.where(image_gray2 / 255 <= 0.5, 1, 0).astype(np.float32)


def draw_opencv():
    """opencv绘制图形"""
    cv2.imshow('image', image)
    cv2.imshow('image_gray1', image_gray)
    cv2.imshow('image_gray2', image_gray2)
    cv2.imshow('image_gray3', image_gray3)
    cv2.imshow('image_binary', image_gray4)
    cv2.imshow('image_binary', image_gray5)
    cv2.imshow('image_binary', image_gray6)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_matplotlib():
    """matplotlib绘制图形"""
    plt.subplot(421)
    plt.imshow(image)
    plt.subplot(423)
    plt.imshow(image_gray, cmap='gray')
    plt.subplot(424)
    plt.imshow(image_gray2, cmap='gray')
    plt.subplot(425)
    plt.imshow(image_gray3, cmap='gray')
    plt.subplot(426)
    plt.imshow(image_gray4, cmap='gray')
    plt.subplot(427)
    plt.imshow(image_gray5, cmap='gray')
    plt.subplot(428)
    plt.imshow(image_gray6, cmap='gray')
    plt.show()


if __name__ == '__main__':
    draw_opencv()
    draw_matplotlib()
    print('test')
