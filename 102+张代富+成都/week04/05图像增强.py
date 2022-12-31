import cv2
import numpy as np


def linear_transform(img, w, b):
    """
    y = w * x + b
    """
    linear_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(linear_img.shape[0]):
        for j in range(linear_img.shape[1]):
            linear_img[i, j] = img[i, j] * w + b
            if linear_img[i, j] > 255:
                linear_img[i, j] = 255
    return linear_img


def log_transform(img, k):
    """
    y = c * log(x+1)
    """
    log_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(log_img.shape[0]):
        for j in range(log_img.shape[1]):
            log_img[i, j] = int(np.log(img[i, j] + 1) * k)
            # if log_img[i, j] > 255:
            #     log_img[i, j] = 255
            # elif log_img[i, j] < 0:
            #     log_img[i, j] = 0
    cv2.normalize(log_img, log_img, 0, 255, cv2.NORM_MINMAX)
    return log_img


def gamma_transform(img, c=1.0, gamma=1.0):
    """
    点处理：伽马变换/幂律变换
    y = c * x^Y
    Y > 1 处理漂白的图片，进行灰度级压缩；反之，处理过黑的图片，对比度增强，突出细节
    """
    gamma_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(gamma_img.shape[0]):
        for j in range(gamma_img.shape[1]):
            gamma_img[i, j] = int(c * img[i, j] ** gamma)
            # if gamma_img[i, j] > 255:
            #     gamma_img[i, j] = 255
            # elif gamma_img[i, j] < 0:
            #     gamma_img[i, j] = 0
        cv2.normalize(gamma_img, gamma_img, 0, 255, cv2.NORM_MINMAX)
        gamma_img = cv2.convertScaleAbs(gamma_img)
    return gamma_img


if __name__ == '__main__':
    image = cv2.imread('./data/lenna.png', 0)
    linear_image = linear_transform(image, 1.1, 1)
    log_image = log_transform(image, 2)
    gamma_image = gamma_transform(image, c=1, gamma=1.2)
    gamma_image2 = gamma_transform(image, c=1, gamma=0.8)
    cv2.imshow('image', image)
    cv2.imshow('linear_img', linear_image)
    cv2.imshow('log_image', log_image)
    cv2.imshow('gamma_image', gamma_image)
    cv2.imshow('gamma_image2', gamma_image2)
    cv2.waitKey(0)
