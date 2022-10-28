# ============================== 1:实现最近邻插值  =================================
import cv2
import numpy as np

def function(img):                       #定义函数
    height,width,channels=img.shape
    #将图像的长，宽，通道信息，赋值给对应的变量

    EmptyImage=np.zeros((1000,1000,channels),np.uint8)
    #uint8是专门用于存储各种图像的（包括RGB，灰度图像等），范围是从0–255
    #建立一个长宽均为1000像素的0矩阵图像   ，为了将原图像进行放大

    New_height=1000/height               #长度进行放大后的值
    New_width=1000/width                 #宽度进行放大后的值

    for i in range(1000):                #开始对横坐标进行遍历
        for j in range(1000):            #开始对纵坐标进行遍历
            x=int(i/New_height+0.5)      #在原图像中插入的坐标的值，+0.5是为了方便进行四舍五入
            y=int(j/New_width+0.5)
            EmptyImage[i,j]=img[x,y]     #循环完后所有新图像的像素值均来自于原图像

    return EmptyImage                    #返回新头

img=cv2.imread("lenna.png")              #导入图片
T=function(img)                          #函数的调用
# print(T)
# print(T.shape)
cv2.imshow("nearest interp",T)           #图像的显示
cv2.imshow("image",img)
cv2.waitKey(0)                           #保持图像的显示





