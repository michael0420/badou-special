import numpy as np
import cv2

img=cv2.imread('lenna.png',2|4)
# img=cv2.imread('lenna.png',0)
# cv2.imshow('img_gray',img)A

Gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('canny',cv2.Canny(img,100,100))
#第一个参数是需要处理的原图像，(单通道的灰度图)；需要进行图像灰度化处理
#第二个参数是滞后阈值1
#第三个参数是滞后阈值2


#laplace算子
# img_laplace = cv2.Laplacian(Gray, cv2.CV_64F, ksize=3)
# cv2.imshow('laplace',img_laplace)
# cv2.imshow('laplace',cv2.Laplacian(Gray,cv2.CV_64F,ksize=3))


cv2.waitKey(0)
#








import numpy as np
import cv2

def CannyThreshold(lowthreshold):                 #定义函数
    detected=cv2.GaussianBlur(gray,(3,3),0)       #（灰度图，卷积核，标准差）
    detected=cv2.Canny(detected,
                       lowthreshold,              #低阈值
                       lowthreshold*ratio,        #高阈值是低阈值的三倍
                       apertureSize=kernel_size   #kernel_size用于设置canny算法内部的sobel边缘提取
                       )
    dst=cv2.bitwise_and(img,img,mask=detected)    #进行遮罩
    #cv2.bitwise_and()是对二进制数据进行“与”操作，即对图像（灰度图像或彩色图像均可）每个像素值
    #进行二进制“与”操作，1&1=1，1&0=0，0&1=0，0&0=0，利用掩膜（mask）进行“与”操作

    cv2.imshow('Canny_demo',dst)

lowthreshold=0
max_lowthreshold=100
ratio=3
kernel_size=3

img=cv2.imread('lenna.png',2|4)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.namedWindow('Canny_demo')
##用于创建一个具有合适名称和大小的窗口，以在屏幕上显示图像和视频。默认情况下，
# 图像以其原始大小显示，因此我们可能需要调整图像大小以使其适合我们的屏幕。
# cv2.namedWindow(window_name, flag)
# window_name：将显示图像/视频的窗口的名称
# flag： 表示窗口大小是自动设置还是可调整。


#进行杠杆调节，处理图像
cv2.createTrackbar('New','Canny_demo',lowthreshold,max_lowthreshold,CannyThreshold)

#cv2.createTrackbar()
# 第一个参数，是这个trackbar对象的名字
# 第二个参数，是这个trackbar对象所在面板的名字
# 第三个参数，是这个trackbar的默认值,也是调节的对象
# 第四个参数，是这个trackbar上调节的范围(0~count)
# 第五个参数，是调节trackbar时调用的回调函数名

CannyThreshold(0)    #初始化
if cv2.waitKey(0)==27:
    cv2.destroyAllWindows()









import numpy as np
import cv2
import matplotlib.pyplot as plt

img=cv2.imread('lenna.png',2|4)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#sobel算子
img_sobel_x=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
img_sobel_y=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)

#laplace算子
dst=cv2.Laplacian(gray,cv2.CV_64F,ksize=3)


#canny算子
New_dst=cv2.Canny(gray,100,100)

#图像显示
plt.subplot(2,3,1)
plt.imshow(gray, "gray"), plt.title("Original")
plt.subplot(2,3,2)
plt.imshow(img_sobel_x,"gray"),plt.title("img_sobel_x")
plt.subplot(2,3,3)
plt.imshow(img_sobel_y,"gray"),plt.title("img_sobel_y")
plt.subplot(2,3,4)
plt.imshow(dst,"gray"),plt.title("Laplacian")
plt.subplot(2,3,5)
plt.imshow(New_dst,"gray"),plt.title("Canny")


plt.show()

