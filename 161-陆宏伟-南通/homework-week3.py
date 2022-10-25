import cv2
import numpy as np

# flags -1 原图 0 灰度图 1：彩图
imgdata = cv2.imread("lenna.png", flags=1)
h, w, channel = imgdata.shape

newH = 200
newW = 200
newimg = np.zeros((200, 200, channel), imgdata.dtype)


# 作业 1 最邻近差值
hrate = h/newH
wrate = w/newW

for i in range(newH):
    for j in range(newW):
        srci = int(i * hrate + 0.5)
        srcj = int(j * wrate + 0.5)
        newimg[i][j] = imgdata[srci][srcj]

cv2.imshow("最邻近差值", newimg)    # 显示图像
cv2.waitKey()
# cv2.destroyAllWindows()


#作业 2 直方图均衡化
b, g, r = cv2.split(imgdata)

newb = cv2.equalizeHist(b)
newg = cv2.equalizeHist(g)
newr = cv2.equalizeHist(r)
# 通道合成方法 第一个参数为元组
newimg = cv2.merge((newb, newg, newr))

cv2.imshow("直方图均衡化", newimg)    # 显示图像
cv2.waitKey()



# 作业 3 双线性插值

dH = 700
dW = 700
dhrate = h/dH
dwate = w/dW

img3 = np.zeros((dH, dW, channel), imgdata.dtype)

for c in range(3):
    for i in range(dH):
        for j in range(dW):
            # print(j)
            # 中心对称
            srci = (i + 0.5) * dhrate - 0.5
            srcj = (j + 0.5) * dwate - 0.5
            srcX0 = int(np.floor(srcj))
            srcX1 = min(srcX0 + 1, w -1)
            srcY0 = int(np.floor(srci))
            srcY1 = min(srcY0 + 1, h -1)
            x = srcj - srcX0
            y = srci - srcY0
            temp0 = x * imgdata[srcY0, srcX1, c] + (1-x) * imgdata[srcY0, srcX0, c]
            temp1 = x * imgdata[srcY1, srcX1, c] + (1-x) * imgdata[srcY1, srcX0, c]
            img3[i, j, c] = int(y * temp1 + (1-y) * temp0)
            # print("111")

cv2.imshow("双线性插值", img3)    # 显示图像
cv2.waitKey()
cv2.destroyAllWindows()








