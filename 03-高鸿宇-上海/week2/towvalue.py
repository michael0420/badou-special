import cv2 as cv
import numpy as np

def myTwoValue(img):
    img = np.where(img <= 0.5, 0, 1).astype(np.float32)
        
    cv.imshow("lenna", img)
    cv.waitKey()

if __name__ == "__main__":
    img = cv.imread(r'week2\lenna.png', cv.IMREAD_GRAYSCALE)
    img = img / 255
    myTwoValue(img)