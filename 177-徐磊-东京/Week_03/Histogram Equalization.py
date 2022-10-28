import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
Histogram Equalization
'''

img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray_hist_equ = cv2.equalizeHist(gray)

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
plt.hist(gray.ravel(), 256)
plt.title('Original Histogram')

ax2 = fig.add_subplot(2, 2, 2)
plt.hist(gray_hist_equ.ravel(), 256)
plt.title('Histogram Equalization')

ax3 = fig.add_subplot(2,2,3)
plt.imshow(gray, cmap='gray')
plt.title("Gray Image")

ax4 = fig.add_subplot(2,2,4)
plt.imshow(gray_hist_equ, cmap='gray')
plt.title('Hist Equalization Image')

fig.subplots_adjust(hspace=0.5, wspace=0.5)
fig.show()
fig.savefig('Histogram Equalization.png')

