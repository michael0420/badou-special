"""
canny算法open-cv接口,增加可选阈值按钮
"""

import cv2

image = cv2.imread('./data/lenna.png')
image = cv2.resize(image, (1000, 800), interpolation=cv2.INTER_AREA)


def canny(threshold, img=image, ratio=2.5, kernel_size=3):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussian_img = cv2.GaussianBlur(img_gray, (3, 3), 0)
    edges = cv2.Canny(gaussian_img, threshold, threshold * ratio, apertureSize=kernel_size)
    cv2.imshow('canny', cv2.bitwise_or(img, img, mask=edges))


if __name__ == '__main__':
    cv2.namedWindow('canny')
    cv2.createTrackbar('low threshold', 'canny', 0, 120, canny)
    canny(0, image)
    cv2.waitKey()
    cv2.destroyAllWindows()
