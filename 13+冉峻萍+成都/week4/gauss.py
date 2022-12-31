import cv2
import random

def gauss_noise(src, means, sigma, percetage):
    NoiseImg = src
    NoiseNum = int(percetage*src.shape[0]*src.shape[1])
    
    # 遍历图像中的像素点，噪声点个数是根据 percetage 和 图像的长、宽计算出来的
    for i in range(NoiseNum):
        # 随机生成坐标
        randX = random.randint(0, src.shape[0]-1)
        randY = random.randint(0, src.shape[1]-1)
        
        # 原图像像素灰度值上加上高斯随机数
        NoiseImg[randX, randY] = NoiseImg[randX, randY] + random.gauss(means, sigma)
        
        # 图像灰度像素值缩放到 0~255 之间
        if NoiseImg[randX, randY] < 0:
            NoiseImg[randX, randY] = 0
        elif NoiseImg[randX, randY] > 255:
            NoiseImg[randX, randY] = 255
            
    return NoiseImg

img = cv2.imread("Lenna.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 注意，这里 img_gray 会被修改，所以要显示原始图像灰度图，需要重新读取
img_gauss = gauss_noise(img_gray, 2, 4, 0.8)

img = cv2.imread("Lenna.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("img_gauss", img_gauss)
cv2.imshow("img_gray", img_gray)

cv2.waitKey(0)

