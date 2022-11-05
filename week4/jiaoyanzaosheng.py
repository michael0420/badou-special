import cv2
import random
def jiaoyan(img,percent):
    w,h=img.shape
    for i in range(int(w*h*percent)):
        x=random.randint(0,w-1)
        y=random.randint(0,h-1)
        if random.random()>0.5:
            img[x,y]=0
        else:
            img[x,y]=255
    return img
img=cv2.imread("lenna.png")
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
jiaoyan_img=jiaoyan(img,0.1)
cv2.imshow('jiaoyan',jiaoyan_img)
cv2.waitKey()