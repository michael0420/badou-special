import cv2
import random

# 实现高斯噪声
def gaussian_noise(img, means, sigma, percentage):
    noise_img = img
    noise_num = int(percentage * img.shape[0] * img.shape[1])

    for i in range(noise_num):
        rand_x = random.randint(0, img.shape[1] - 1)
        rand_y = random.randint(0, img.shape[0] - 1)
        # 在原有像素值上加一个高斯的随机数
        noise_img[rand_y, rand_x] = img[rand_y, rand_x] + random.gauss(means, sigma)

        if noise_img[rand_y, rand_x] < 0:
            noise_img[rand_y, rand_x] = 0
        elif noise_img[rand_y, rand_x] > 255:
            noise_img[rand_y, rand_x] = 255

    return noise_img


if __name__ == '__main__':
    img = cv2.imread(r'D:\JetBrainsProjects\PycharmProjects\CV\lenna.png', 0)
    cv2.imshow("src", img)
    img2 = gaussian_noise(img, 9, 9, 0.97)
    cv2.imshow("gaussian noise", img2)
    cv2.waitKey()
