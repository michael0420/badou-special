import cv2
import matplotlib.pyplot as plt

img = cv2.imread("lenna.png")
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# cv2.imshow("gray", gray)
# print(gray)
# cv2.waitKey()


# plt.figure()
# plt.hist(gray.ravel(),256)
# plt.show()
img_hist = cv2.calcHist(gray,[0],None,[256],[0,255])
plt.figure()
plt.title("histogram")
plt.xlabel("x")
plt.ylabel("num")
plt.xlim(0, 255)
plt.plot(img_hist)
plt.show()

# cv2.waitKey(0)
#彩色的
image = cv2.imread("lenna.png")
cv2.imshow("Original", image)
chans = cv2.split(image)
colors = ("b", "g", "r")
plt.figure()
plt.title("Flattened Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

for (chan,color) in zip(chans,colors):
    hist = cv2.calcHist([chan],[0],None,[256],[0,256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])
plt.show()



#equalhistogram
img = cv2.imread("lenna.png")
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
equalHi = cv2.equalizeHist(gray)
plt.figure()
plt.hist(equalHi.ravel(),256)
plt.show()
cv2.imshow("equalHistogram",equalHi)


#彩色的equalhistogram
img = cv2.imread("lenna.png")
cv2.imshow("first original", img)
(b,g,r) = cv2.split(img)
Hb=cv2.equalizeHist(b)
Hg=cv2.equalizeHist(g)
Hr=cv2.equalizeHist(r)
result = cv2.merge((Hb, Hg, Hr))
cv2.imshow("change ",result)
cv2.waitKey()