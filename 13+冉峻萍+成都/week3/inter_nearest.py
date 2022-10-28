import cv2
import numpy as np

def inter_nearest(img, Nh, Nw):
    height,width,channels = img.shape
    Ni = np.zeros((Nh, Nw, channels), np.uint8)
    for i in range(Nh):
        for j in range(Nw):
            x = int(i*(height/Nh) + 0.5)
            y = int(j*(width/Nw) + 0.5)
            Ni[i,j] = img[x,y]
            
    return Ni

img = cv2.imread("Lenna.png")
Ni = inter_nearest(img, 800, 800)
cv2.imshow("Nearest", Ni)
cv2.imshow("Origin", img)

cv2.waitKey(0)