
import cv2 as cv
import numpy as np
import sys

def rgb2gray(img):
    '''
    手动将RGB图像转换为gray
    :param img: RGB图像
    :return: 灰度图像
    '''
    h, w = img.shape[:2]  #h是高度，w是宽度
    b, g, r = cv.split(img)  #通道分离
    gray_img = np.zeros((h,w), dtype='uint8')

    for i in range(0,h):
        for j in range(0, w):
            gray_img[i][j] = 0.11 * b[i][j] + 0.59 * g[i][j] + 0.3 * r[i][j]
    return gray_img

def my_thres(gray_img):
    '''
    将灰度图像二值化
    :param gra_img: 灰度图像
    :return: 二维画的图像
    '''
    print("gray_img: ",gray_img)
    print("np.where result:", np.where(gray_img>=125, 255, 0))
    ret_img = np.where(gray_img>=125, 255, 0)
    return ret_img.astype('uint8')    #图像的处理是无符号八位整型，需进行类型转换

if __name__ == '__main__':
    #读取图像并判断是否读取成功
    img = cv.imread('lena.jpg')
    if img is None :
        print('Failed to read lena.jpg')
        sys.exit()
    else:
        cv.imshow('origin image ', img)

        #调用自己的实现函数 转为灰度图像
        my_gray_img = rgb2gray(img)
        # 调用cv库的函数 转为灰度图像
        gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # 调用自己的实现函数 将灰度图像进行二值化
        my_thres_img = my_thres(my_gray_img)
        # 调用cv库将灰度图像二值化
        _, thres_img = cv.threshold(gray_image, 125, 255, cv.THRESH_BINARY)

        cv.imshow('gray image', gray_image)
        cv.imshow('my_gray_img', my_gray_img)
        cv.imshow('thres image', thres_img)
        print("my_thres_img:", my_thres_img)
        cv.imshow('my_thres_img', my_thres_img)

        cv.waitKey(0)
        cv.destroyAllWindows()
