#双线性插值，放大到1000*1000
import cv2
import numpy as np

img = cv2.imread('lenna.png')
H,W,channels = img.shape      #读取宽，长，通道数
new_img = np.zeros((1000,1000,channels),np.uint8) #创建一张新图片
for chan in range(0,channels):
    for dst_x in range(0,1000):
        for dst_y in range(0,1000):
            #中心对齐,公式中的x,y
            src_x = (dst_x+0.5)*(W/1000)-0.5
            src_y = (dst_y+0.5)*(H/1000)-0.5

            #找到对应的点,x1,x2,y1,y2
            src_x1 = int(np.floor(src_x))
            src_x2 = min(src_x1+1, W-1)

            src_y1 = int(np.floor(src_y))
            src_y2 = min(src_y1+1, H - 1)
            #公式，可以通过查看img的存储形式查看使用dst_x,dst_y,chan进行索引
            temp1 = (src_x2-src_x)*img[src_x1,src_y1,chan]+(src_x-src_x1)*img[src_x2,src_y1,chan]
            temp2 = (src_x2-src_x)*img[src_x1,src_y2,chan]+(src_x-src_x1)*img[src_x2,src_y2,chan]
            new_img[dst_x,dst_y,chan] = (src_y2-src_y)*temp1+(src_y-src_y1)*temp2

cv2.imshow('bilinear interp',new_img)
cv2.waitKey(0)