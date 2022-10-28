#直方图均衡化
import cv2
import numpy as np
#img = np.array([[1,1,2],[0,1,2],[1,2,3]])
from matplotlib.pyplot import subplot
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import matplotlib
matplotlib.rc("font", family='YaHei Consolas Hybrid')

img = cv2.imread('2.jpg')
img2 = img.copy()
(b,g,r) = cv2.split(img) # 将b,g,r分开
img_b = b.copy()
img_g = g.copy()
img_r = r.copy()
H,W = img.shape[:2]
S = H*W   #总像素数
temp_b = 0
temp_g = 0
temp_r = 0
for i in range(0,256):
    axis_b = np.where(b==i)  #像素值为i的坐标,x=[],y=[]
    temp_b += len(img_b[axis_b])  #修改处
    sumPi_b = float(temp_b/S)
    img_b[axis_b] = (sumPi_b*256-1+0.5)
for i in range(0, 256):
    axis_g = np.where(g == i)  # 像素值为i的坐标,x=[],y=[]
    temp_g += len(img_g[axis_g])  #修改处
    sumPi_g = float(temp_g / S)
    img_g[axis_g] = (sumPi_g * 256 - 1 + 0.5)
for i in range(0, 256):
    axis_r = np.where(r == i)  # 像素值为i的坐标,x=[],y=[]
    temp_r += len(img_r[axis_r])  #修改处
    sumPi_r = float(temp_r / S)
    img_r[axis_r] = (sumPi_r * 256 - 1 + 0.5)

img2 = cv2.merge((img_r, img_g, img_b))
img3 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   #将均衡化的r,g,b合并成新图像

subplot(121)
plt.imshow(img3)
plt.title('原图')

subplot(122)
plt.imshow(img2)
plt.title('处理后')
plt.show()

