import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def gray_equalizeHist(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dst = cv.equalizeHist(gray)
    dst_hist = cv.calcHist([dst], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
    cv.imshow("Histogram Equalization", np.hstack([gray, dst]))
    plt.figure()
    plt.xlabel('Bins')
    plt.ylabel('number of pixels')
    plt.plot(dst_hist)
    plt.xlim([0, 256])
    plt.show()

def color_equalizeHist(img):
    (b, g, r) = cv.split(img)
    bH = cv.equalizeHist(b)
    gH = cv.equalizeHist(g)
    rH = cv.equalizeHist(r)
    dst_color = cv.merge((bH, gH, rH))
    cv.imshow(" Color Histogram Equalization", np.hstack([img, dst_color]))
    colors = ('b', 'g', 'r')
    plt.figure()
    plt.xlabel('Bins')
    plt.ylabel('number of pixels')
    for i, col in enumerate(colors):
        hist = cv.calcHist([dst_color], channels=[i], mask=None, histSize=[256], ranges=[0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.show()

if __name__ == '__main__':
    img = cv.imread("lenna.png")
    gray_equalizeHist(img)
    color_equalizeHist(img)
    cv.waitKey(0)