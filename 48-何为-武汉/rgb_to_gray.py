import numpy as np
import cv2 as cv


class Rgb2Gray:

    def __init__(self, img_path):
        self.img_path = img_path
        self.image = cv.imread(self.img_path)
        pass

    # 调用 cv库
    def rgb_to_gray_cv(self):
        out_image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        cv.imwrite("gray_cv.jpg", out_image)

    # cv 库读取 是 BGR

    # 浮点方法
    def rgb_to_gray_v1(self):
        out_image = self.image[:, :, 0] * 0.11 + self.image[:, :, 1] * 0.59 + self.image[:, :, 2] * 0.3
        cv.imwrite("gray_V1.jpg", out_image)

    # 移位方法
    def rgb_to_gray_v2(self):
        h,w = self.image.shape[:2]
        out_image = np.zeros((h,w),self.image.dtype)
        for i in range(h):
            for j in range(w):
                out_image[i,j]=(self.image[i, j, 0] * 28 + self.image[i, j, 1] * 151 + self.image[i, j, 2] * 76)>>8

        cv.imwrite("gray_V2.jpg", out_image)

    # 平均值法
    def rgb_to_gray_v3(self):
        out_image = self.image[:, :, 0]/3 + self.image[:, :, 1]/3 + self.image[:, :, 2] / 3
        cv.imwrite("gray_V3.jpg", out_image)

    # 仅取绿色
    def rgb_to_gray_v4(self):
        out_image = self.image[:, :, 1]
        cv.imwrite("gray_V4.jpg", out_image)


if __name__ == '__main__':
    rgb_to_gray = Rgb2Gray('test.jpeg')
    rgb_to_gray.rgb_to_gray_cv()
    rgb_to_gray.rgb_to_gray_v1()
    rgb_to_gray.rgb_to_gray_v2()
    rgb_to_gray.rgb_to_gray_v3()
    rgb_to_gray.rgb_to_gray_v4()
