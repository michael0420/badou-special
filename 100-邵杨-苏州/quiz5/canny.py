import cv2 as cv
import numpy as np

def blur(img):
    blur = cv.GaussianBlur(img, (5, 5), cv.BORDER_DEFAULT)
    return blur

def threshold(lowthresh):
    gaussian_img = blur(gray)
    detected_edges = cv.Canny(gaussian_img, lowthresh, lowthresh * ratio, apertureSize=3)
    dst = cv.bitwise_and(img, img, mask=detected_edges)

    contours, hierarchies = cv.findContours(cv.Canny(gaussian_img, lowthresh, lowthresh * ratio, apertureSize=3),
                                            mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_SIMPLE)
    #print(f'{len(contours)} contours found')
    blank = np.zeros(img.shape, dtype='uint8')
    cv.drawContours(blank, contours, contourIdx=-1, color=(255, 255, 255), thickness=1)
    cv.imshow("canny demo and edges and origin", np.hstack([dst, blank, img]))


if __name__ == '__main__':
    ratio = 4
    lowthresh = 0
    max_lowthresh = 100
    img = cv.imread('lenna.png')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.namedWindow('canny demo and edges and origin')
    cv.createTrackbar('Min threshold', 'canny demo and edges and origin', lowthresh, max_lowthresh, threshold)
    threshold(0)
    cv.waitKey(0)