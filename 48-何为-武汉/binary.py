import numpy as np
import cv2 as cv

class Binary:
    def __init__(self, img_path):
        self.img_path = img_path
        self.image = cv.imread(self.img_path)
        self.gray_image = None
        pass

    # 调用 cv库
    def rgb_to_gray_cv(self):
        self.gray_image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)


    def gray_to_binary(self):
        self.rgb_to_gray_cv()
        out_image = np.where(self.gray_image<128,0,255)
        cv.imwrite("binary.jpg", out_image)

if __name__ == '__main__':
    binary = Binary('test.jpeg')
    binary.gray_to_binary()