import cv2 as cv
import numpy as np


if __name__ == '__main__':
    img = cv.imread(r'week5\lenna.png')
    edge = cv.Canny(img, 100, 300)
    cv.imshow("lenna", edge)
    cv.waitKey(0)
