import cv2
import random


def gaussian_noise(img, mu, sigma, rate):
    noise_img = img.copy()
    noise_num = int(img.shape[0] * img.shape[1] * rate)
    for i in range(noise_num):
        rand_x = random.randint(0, img.shape[0] - 1)
        rand_y = random.randint(0, img.shape[1] - 1)
        noise_img[rand_x, rand_y] = noise_img[rand_x, rand_y] + random.gauss(mu, sigma)
        if noise_img[rand_x, rand_y] < 0:
            noise_img[rand_x, rand_y] = 0
        elif noise_img[rand_x, rand_y] > 255:
            noise_img[rand_x, rand_y] = 255
    return noise_img


if __name__ == '__main__':
    image = cv2.imread('../data/lenna.png', 0)
    noise_image = gaussian_noise(image, 2, 4, 0.8)
    cv2.imshow("image", image)
    cv2.imshow("noise_image", noise_image)
    cv2.waitKey(0)
