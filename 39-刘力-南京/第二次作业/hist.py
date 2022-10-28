import cv2
import matplotlib.pyplot as plt


# 灰度图的直方图
img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('gray', img)
cv2.waitKey()


# 方法一
plt.figure()
plt.hist(img.ravel(), 256)
plt.show()


# 方法二
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.figure()
plt.title('gray_hist')
plt.xlabel('bins')
plt.ylabel('counts')
plt.xlim([0, 256])
plt.plot(hist)
plt.show()

# 彩色图直方图
img = cv2.imread('lenna.png')
colors = ['b', 'g', 'r']
chans = cv2.split(img)
plt.figure()
plt.title('color_hist')
plt.xlabel('bins')
plt.ylabel('counts')

for (color, chan) in zip(colors, chans):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])

plt.show()


