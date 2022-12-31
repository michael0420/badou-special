import cv2
import random

# 实现椒盐噪声
def salt_noise(img, percentage):
    noise_img = img
    noise_num = int(percentage * img.shape[0] * img.shape[1])

    for i in range(noise_num):
        rand_y = random.randint(0, img.shape[0] - 1)
        rand_x = random.randint(0, img.shape[1] - 1)
        # 随机数小于0.5得到的像素点直接赋0，否则赋255
        if random.random() <= 0.5:
            noise_img[rand_y, rand_x] = 0
        else:
            noise_img[rand_y, rand_x] = 255

    return noise_img


if __name__ == '__main__':
    img = cv2.imread(r'D:\JetBrainsProjects\PycharmProjects\CV\lenna.png', 0)
    cv2.imshow("src", img)
    img2 = salt_noise(img, 0.05)
    cv2.imshow("salt noise", img2)

    cv2.waitKey()

