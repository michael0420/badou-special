#=============================1:高斯噪声的实现========================

#随机生成符合正态（高斯）分布的随机数，means,sigma为两个参数
import numpy as np
import cv2
import random

#实现图像的灰度化处理
def G_N_Gray(img):
    h=img.shape[0]
    w=img.shape[1]
    Gauss_Img=np.zeros((h,w),np.uint8)
    for i in range(h):
        for j in range(w):
            Gauss_Img[i,j]=0.11*img[i,j,0]+0.59*img[i,j,1]+0.3*img[i,j,2]
            # 图像通道排列是BGR，不是RGB
            # 将RGB图像转化为灰度图的公式
    return Gauss_Img

def Gauss_Noise(img,mean,sigma,per):
    H=img.shape[0]
    W=img.shape[1]
    New_IMG=np.copy(img)                 #图像有效区域的复制
    NP=int(per*H*W)                      #总像素点的个数
    for a in range(NP):
        randX=np.random.randint(1,H-1)   #高斯噪声图片边缘不处理，故-1
        randY=np.random.randint(1,W-1)
        New_IMG[randX,randY]=New_IMG[randX,randY]+random.gauss(mean,sigma)

        if New_IMG[randX,randY]<0:       #将小于等于0的值强制变为0
            New_IMG=[randX,randY]==0
        if New_IMG[randX,randY]>255:     #将大于等于255的值强制变为255
            New_IMG[randX,randY]==255

    return New_IMG

img=cv2.imread('lenna.png')
cv2.imshow('lenna',img)
New_Gray=G_N_Gray(img)
cv2.imshow('New_Gray',New_Gray)
New_Gauss=Gauss_Noise(New_Gray,5,4,0.9)
cv2.imshow('New_Gauss',New_Gauss)
cv2.imwrite('New_Gauss.png',New_Gauss)   #保存图像
cv2.waitKey(0)                           #使图像长时间的显示

