import numpy as np
import cv2


# 图像灰度
class Img2Gray:
    def imagegray(self, image):
        img = cv2.imread(image)
        h, w = img.shape[:2]
        img_gray = np.zeros([h, w], img.dtype)
        for i in range(w):
            for j in range(h):
                m = img[i, j]
                # img_gray[i, j] = m[0]*0.144 + m[1]*0.587 + m[2]*0.299  #加权平均灰度
                # img_gray[i, j] = (m[0] + m[1] + m[2])/3  # 平均灰度
                # img_gray[i, j] = max(m[0], m[1], m[2])  # 最大值灰度
                img_gray[i, j] = m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3  # 浮点灰度
                # img_gray[i, j] = m[0] * 77 + m[1] * 151 + m[2] * 28 >> 8

        cv2.imshow("img gray", img_gray)
        cv2.imshow("img", img)
        cv2.waitKey()
        cv2.destroyAllWindows()


cc = Img2Gray()
cc.imagegray("../images/girl.png")


# 二值化处理
img = cv2.imread("../images/girl.png")
# 调用接口先灰度处理
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 调用接口先灰度处理
h, w = grayImage.shape[:2]
for i in range(h):
    for j in range(w):
        if grayImage[i, j] <= 160:
            grayImage[i, j] = 170
        else:
            grayImage[i, j] = 240


# 调用接口二值化
cv2.imshow("grayImage", grayImage)
cv2.waitKey()

