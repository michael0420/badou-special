#=================================2:椒盐噪声的实现============================

import numpy as np
import cv2
import random

#实现图像的灰度化处理
def Gray(img):
	H=img.shape[0]
	W=img.shape[1]
	img_noise=np.zeros((H,W),np.uint8)
	for a in range(H):
		for b in range(W):
			img_noise[a, b] = 0.11 * img[a,b, 0] + 0.59 * img[a,b, 1] + 0.3 * img[a, b, 2]
			#图像通道排列是BGR，不是RGB
			#将RGB图像转化为灰度图的公式
	return img_noise

def salt_pepper_noise(img,snr):  		#snr为信噪比
	H=img.shape[0]
	W=img.shape[1]
	sp=H*W						 		#总像素点的个数
	img_noise=img.copy()				#图像有效区域的复制
	NP=int((1-snr)*sp)			 		#噪声点的数目
	for i in range(NP):
		X=np.random.randint(1,H-1)		#椒盐噪声图片边缘不处理，故-1
		Y=np.random.randint(1,W-1)

		if random.random()<=0.5:		#random.random生成随机浮点数
			img_noise[X,Y]=0			#将小于等于0.5的值强制变为0
		else:
			img_noise[X,Y]=255			#将大于0.5的值强制变为255，实现椒盐噪声
	return img_noise

img=cv2.imread('lenna.png')             #('lenna.png',0)使图像转化为灰度图
IMG_Gray=Gray(img)
IMG_NOISE=salt_pepper_noise(IMG_Gray,0.7)
cv2.imwrite('New_S_P_N.png',IMG_NOISE)	#保存图像
cv2.imshow('img',img)					#显示原图像
cv2.imshow('gray_p',IMG_Gray)
cv2.imshow('New_img',IMG_NOISE)			#显示添加噪声后的新图片
cv2.waitKey(0)							#使图像长时间的显示


