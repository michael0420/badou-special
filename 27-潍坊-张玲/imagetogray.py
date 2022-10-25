from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 手动灰度化
img = cv2.imread("lenna.png")#注意图片的格式，像素为0-255之间
h,w = img.shape[:2]             #获取彩色图片的高、宽，并且赋值给h和w，这里是单通道；
img_gary = np.zeros([h, w], img.dtype)  # 创建一张和当前图片大小一样的单通道图片只有高、宽，通道为0
for i in range(h):
    for j in range(w):
        m = img[i, j]  # 取出当前高和宽中的BGR坐标
        img_gary[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # 将BGR坐标转化为gray坐标并赋值给新图像
print("image show gray: %s" % img_gary) #注意print的使用方式
cv2.namedWindow("image show gray") #效果没有受到影响
cv2.imshow("image show gray", img_gary) #cv2的imshow必须有第一参数，而且以窗口形式显示
plt.subplot(221)   #图像显示位置
img = plt.imread("lenna.png")#对于PNG图片，读取的图片像素为（0-1）之间的数
plt.imshow(img)    #注意与cv2.imshow的不同
print("-----image lenna-----")
print(img) #打印出来的是图片的像素值

#imag=Image.open("lenna.png")
#imag.show() 另一种图像打开方式

#img = cv2.imread("lenna.png")
#cv2.namedWindow("img_gray")
#img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#注意这里像素值还为0-255之间，所以二值化的时候不能用>=0.5，要用>=120
#cv2.imshow("img_gray",img_gray)
# 调用接口灰度化
img_gray = rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray') #注意用法
print("-----image_gray-----")
print(img_gray)

# 二值化
img_binary = np.where(img_gray >= 0.5, 1, 0) #二值化这里一定注意与使用cvtColor实现灰度化的区别
print("-----image_binary-----")
print(img_binary) #注意print函数的使用
print(img_binary.shape)
plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()  #词条语句很重要
