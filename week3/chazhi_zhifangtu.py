import cv2
import numpy as np

def nearest_interp(img,newsize):
    w,h,c=img.shape
    new_w=newsize[0]
    new_h=newsize[1]
    sw=w/new_w
    sh=h/new_h
    new_img=np.zeros([new_w,new_h,c],img.dtype)
    for i in range(new_w):
        for j in range(new_h):
            x=int(i*sw+0.5)
            y=int(j*sh+0.5)
            new_img[i,j]=img[x,y]
    return new_img

def shaungxianxingchazhi(img,size):
    w,h,c=img.shape
    new_w=size[0]
    new_h=size[1]
    sw=w/new_w
    sh=h/new_h
    new_img=np.zeros([new_w,new_h,c],img.dtype)
    for i in range(new_w):
        for j in range(new_h):
            x=(i+0.5)*sw-0.5
            y=(j+0.5)*sh-0.5
            leftx=int(x)
            lefty=int(y)
            p0=(leftx+1-x)*(lefty+1-y)*img[leftx,lefty]
            p1=(x-leftx)*(lefty+1-y)*img[min(leftx+1,w-1),lefty]
            p2=(leftx+1-x)*(y-lefty)*img[leftx,min(lefty+1,h-1)]
            p3=(x-leftx)*(y-lefty)*img[min(w-1,leftx+1),min(lefty+1,h-1)]
            new_img[i,j]=np.rint(p0+p1+p2+p3)
    return new_img

def zhifangtujunhenghua(img):
    if len(img.shape)==2:
        new_img=cv2.equalizeHist(img)
        return new_img
    else:
        b,g,r=cv2.split(img)
        bh=cv2.equalizeHist(b)
        gh=cv2.equalizeHist(g)
        rh=cv2.equalizeHist(r)
        new_img=cv2.merge((bh,gh,rh))
        return new_img

img=cv2.imread('lenna.png')
img1=nearest_interp(img,[300,300])
cv2.imshow("nearest_interp",img1)
img2=shaungxianxingchazhi(img,[300,300])
cv2.imshow("shaungxianxingchazhi",img2)
img3=zhifangtujunhenghua(img)
cv2.imshow('zhifangtujunhenghua',img3)
cv2.waitKey()