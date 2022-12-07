import cv2 as cv
import numpy as np

def myKmeans(img):
    n_classes = 4
    criteria = (cv.TermCriteria_EPS+cv.TERM_CRITERIA_MAX_ITER, 10, 0.01)
    flags = cv.KMEANS_PP_CENTERS
    src = img.reshape((-1, 3))
    src = np.float32(src)
    
    _, labels, center = cv.kmeans(src, n_classes, None, criteria, 10, flags)
    
    center = np.uint8(center)
    res = center[labels.flatten()]
    dst = res.reshape(img.shape)
    return dst

if __name__ == "__main__":
    img = cv.imread(r'week6\lenna.png')
    src = myKmeans(img)
    cv.imshow('kmeans', np.concatenate((img, src), 1))
    cv.waitKey(0)