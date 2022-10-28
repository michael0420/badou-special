import numpy as np
import cv2 as cv

if __name__ == "__main__":
   ori = cv.imread(r'week2\lenna.png', cv.IMREAD_GRAYSCALE)
   dst = cv.equalizeHist(ori)
   cv.imshow("lenna", np.concatenate((ori, dst), 1))
   cv.waitKey(0)