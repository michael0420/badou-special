import cv2
import random


def pepper_salt_noise(img, rate):
    noise_img = img.copy()
    noise_num = int(img.shape[0] * img.shape[1] * rate)
    for i in range(noise_num):
        rand_x = random.randint(0, img.shape[0] - 1)
        rand_y = random.randint(0, img.shape[1] - 1)

        if random.random() <= 0.5:
            noise_img[rand_x, rand_y] = 0
        elif random.random() > 0.5:
            noise_img[rand_x, rand_y] = 255
    return noise_img


if __name__ == '__main__':
    image = cv2.imread('./data/lenna.png', 0)
    noise_image = pepper_salt_noise(image, 0.02)
    cv2.imshow("image", image)
    cv2.imshow("noise_image", noise_image)
    cv2.waitKey(0)
