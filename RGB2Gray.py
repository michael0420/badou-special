#彩色图像灰度化

# import cv2
# import numpy as np

# img = cv2.imread('lenna.png')
# H,W = img.shape[:2]
# img_gray = np.zeros([H,W],img.dtype)#生成一个与原图大小相同，单通道，像素值为0的图片
# #X = img_gray.shape[:3] 读取图片的规格信息（大小、通道数）
# #print(X)
# for i in range(H):
#     for j in range(W):
#         img_P = img[i,j]
#         img_gray[i,j] = int(img_P[0]*0.11 + img_P[1]*0.59 + img_P[2]*0.30)#opencv读取的彩色图像的通道顺序是BGR真实彩色图片通道数是RGB  Gray = R*0.30 + G*0.59 + B*0.11（RGB2Gray）


# cv2.imshow('lenna',img)
# cv2.imshow('img_gray',img_gray)
# cv2.waitKey(0)

#图像二值化
# import cv2

# img  = cv2.imread('lenna.png')
# img_G = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#转化为灰度图
# H,W = img_G.shape[:2]#读取图像的长宽

# #遍历图像像素点
# for i in range(H):
#     for j in range(W):
#         img_P = img_G[i,j]
#         #print(img_P)
#         #判断像素点的大小
#         if img_P <= 178:
#             img_P = 0
#         elif img_P > 178:
#             img_P = 255
# cv2.imshow('orgnial',img_G)
# cv2.imshow('result',img_P)
# cv2.waitKey()
import cv2
img = cv2.imread('lenna.png',cv2.IMREAD_UNCHANGED)
print(img.shape)
img1,img2,img3 = cv2.split(img)
cv2.imshow('orginal',img)
cv2.imshow('B',img1)
cv2.imshow('G',img2)
cv2.imshow('R',img3)
cv2.waitKey()

