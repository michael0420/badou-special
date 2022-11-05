import cv2
img=cv2.imread("lenna.png")
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img=cv2.Canny(img,150,250)
cv2.imshow("canny",img)
cv2.waitKey()