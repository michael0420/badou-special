import cv2
import random

def gauss(img,percent,mu,sigma):
    w,h=img.shape
    for i in range(int(w*h*percent)):
        x=random.randint(0,w-1)
        y=random.randint(0,h-1)
        a=img[x,y]+random.gauss(mu,sigma)
        if a>255:
            img[x,y]=255
        elif a<0:
            img[x,y]=0
        else:
            img[x,y]=a
    return img

img=cv2.imread("lenna.png")
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gauss_img=gauss(img,0.5,0,10)
cv2.imshow("gauss",gauss_img)
cv2.waitKey()