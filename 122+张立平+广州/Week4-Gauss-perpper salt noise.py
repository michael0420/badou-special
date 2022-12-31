import cv2
import numpy as np
from numpy import shape
import random

'''
Gauss noise
Pout = Pin + random.gauss
random.gauss是通过sigma和mean生成符合高斯分布的随机数，并且需要指定噪声处理范围
'''
def Gaussnoise(src, sigma, mean, percentage):
    dst = src
    NoiseNum = int(percentage * src.shape[0] * src.shape[1])

    for i in range(NoiseNum):
        # 高斯噪声图片边缘不处理，故-1
        ranx = random.randint(0,src.shape[0]-1)
        rany = random.randint(0, src.shape[1] - 1)
        dst[ranx,rany] = src[ranx,rany] + random.gauss(mean,sigma)
        if dst[ranx,rany]<0:
            dst[ranx,rany]=0
        elif dst[ranx,rany]>255:
            dst[ranx,rany]=255
    return dst

Ori= cv2.cvtColor(cv2.imread("lenna.png"),cv2.COLOR_BGR2GRAY)
src = cv2.imread("lenna.png",0)
dst = Gaussnoise(src,2,3,1)
cv2.imshow("Gaussnoise",np.hstack([Ori,dst]))
cv2.waitKey()

'''
为什么如果我用cv2.imshow("Gaussnoise",np.hstack([src,dst]))
显示的原图也是加噪修改后的dst呢？
'''


'''
Pepper - Salt noise
指定SNR，[0,1]
要加噪的像数数目 NP = 总像数数目 SP * SNR
随机获取加噪位置
指定像素值为0/255
'''
def peppersalt(src,SNR):
    dst = src
    SP = src.shape[0] * src.shape[1]
    NP = int(SP * SNR)
    for i in range(NP):
        ranx = random.randint(0,src.shape[0]-1)
        rany = random.randint(0,src.shape[1]-1)
        if src[ranx,rany]<=128:
            dst[ranx,rany]=0
        else:
            dst[ranx, rany] = 255
    return dst

Ori= cv2.cvtColor(cv2.imread("lenna.png"),cv2.COLOR_BGR2GRAY)
src = cv2.imread("lenna.png",0)
dst = peppersalt(src,0.7)
cv2.imshow("Pepper-Salt",np.hstack([Ori,dst]))
cv2.waitKey()