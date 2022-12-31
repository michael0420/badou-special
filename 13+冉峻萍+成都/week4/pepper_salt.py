import cv2
import random

def pepper_salt_noise(src, percetage):
    NoiseImg = src
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    
    for i in range(NoiseNum):
        randX = random.randint(0, src.shape[0]-1)
        randY = random.randint(0, src.shape[1]-1)
        
        if random.random() <= 0.5:
            NoiseImg[randX,randY] = 0
        else:
            NoiseImg[randX,randY] = 255
    
    return NoiseImg


img = cv2.imread("Lenna.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_pepper_salt = pepper_salt_noise(img_gray, 0.2)

img = cv2.imread("Lenna.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("img_pepper_salt", img_pepper_salt)
cv2.imshow("img_gray", img_gray)

cv2.waitKey(0)