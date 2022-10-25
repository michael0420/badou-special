#-*- coding: utf-8 -*-

from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
import cv2

#
# 灰度化
# 读照片-获取照片大小-创建同大小空白矩阵-给空白矩阵对应单元格灰度化并赋值-输出矩阵与图片
Img=cv2.imread("lenna.png")
h,w=Img.shape[:2]
Img_gray=np.zeros([h,w],Img.dtype)
for i in range(h):
    for j in range(w):
        m=Img[i,j]
        # R0.3+G0.59+B0.11
        Img_gray[i,j]=int(m[0]*0.11+m[1]*0.59+m[2]*0.3)
print('---Img---\n%s'%Img)
print('---Img_gray---\n%s'%Img_gray)
cv2.imshow("Img_gray",Img_gray)
cv2.waitKey()

plt.subplot(221)
Img=plt.imread("lenna.png")  #有这句输出原图，没有会因为opencv是BGR导致图片颜色不一样
plt.imshow(Img)
Img_gray=rgb2gray(Img)
plt.subplot(222)
plt.imshow(Img_gray,cmap='gray')

#二值化
# 在灰度图基础上设置临界值
# h1,w1=Img.shape[:2] = Img_gray.shape
# Img_binary=Img_gray
# for i in range(h1):
#     for j in range(w1):
#         if Img_binary[i,j] >=0.5:
#             Img_binary[i, j]=1
#         else:
#             Img_binary[i, j] = 0

Img_binary=np.where(Img_gray>=0.5,1,0)
plt.subplot(223)
plt.imshow(Img_binary,cmap='gray')
print('---Img_binary---\n%s'%Img_binary)
print(Img_binary)
plt.show()
