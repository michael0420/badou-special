import cv2

img = cv2.imread("Lenna.jpg", cv2.IMREAD_GRAYSCALE)
        
cv2.imshow("Gray", img)

cv2.waitKey(0)