import cv2
import matplotlib.pyplot as plt

image = cv2.imread('./data/lenna.png', 0)

# sobel
image_sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
image_sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
# laplace
image_laplace = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
# canny
image_canny = cv2.Canny(image, 40, 250)

plt.subplot(231), plt.imshow(image, "gray"), plt.title("Original")
plt.subplot(232), plt.imshow(image_sobel_x, "gray"), plt.title("Sobel_x")
plt.subplot(233), plt.imshow(image_sobel_y, "gray"), plt.title("Sobel_y")
plt.subplot(234), plt.imshow(image_laplace, "gray"), plt.title("Laplace")
plt.subplot(235), plt.imshow(image_canny, "gray"), plt.title("Canny")
plt.show()
