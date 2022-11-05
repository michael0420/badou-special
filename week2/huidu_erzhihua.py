import cv2
import numpy as np

img=cv2.imread('lenna.png')
h,w=img.shape[:2]

#灰度图像
img_fudian_gray=np.zeros([h,w],img.dtype)#浮点算法
img_zhengshu_gray=np.zeros([h,w],img.dtype)#整数方法
img_yiwei_gray=np.zeros([h,w],img.dtype)#移位算法
img_pingjun_gray=np.zeros([h,w],img.dtype)#平均值法
img_lv_gray=np.zeros([h,w],img.dtype)#仅取绿色
for i in range(h):
    for j in range(w):
        m=img[i,j]
        img_fudian_gray[i,j]=int(m[0]*0.11+m[1]*0.59+m[2]*0.3)
        img_zhengshu_gray[i,j]=int((m[0]*11+m[1]*59+m[2]*30)/100)
        img_yiwei_gray[i,j]=int((m[0]*76+m[1]*151+m[2]*28)>>8)
        img_pingjun_gray[i,j]=int(m[0]/3+m[1]/3+m[2]/3)
        img_lv_gray[i,j]=int(m[1])
cv2.imshow("img_fudian_gray",img_fudian_gray)
cv2.imshow("img_zhengshu_gray",img_zhengshu_gray)
cv2.imshow("img_yiwei_gray",img_yiwei_gray)
cv2.imshow("img_pingjun_gray",img_pingjun_gray)
cv2.imshow("img_lv_gray",img_lv_gray)


#二值化
img_erzhihua=np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        if (img_fudian_gray[i,j]/255)>0.5:
            img_erzhihua[i,j]=255
        else:
            img_erzhihua[i,j]=0
cv2.imshow("img_erzhihua",img_erzhihua)
key = cv2.waitKey(0)