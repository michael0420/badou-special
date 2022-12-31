from skimage import util
import cv2

image = cv2.imread('./data/lenna.png')
noise_image1 = util.random_noise(image, mode='gaussian')
noise_image2 = util.random_noise(image, mode='s&p')
# cv2.imwrite('./data/gaussian.png', noise_image1)
# cv2.imwrite('./data/s&p.png', noise_image2)
cv2.imshow('image', image)
cv2.imshow('gaussian', noise_image1)
cv2.imshow('s&p', noise_image2)

cv2.waitKey(0)
