import cv2

img = cv2.imread("Lenna.png")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("canny", cv2.Canny(img_gray, 200, 300))

cv2.waitKey(0)