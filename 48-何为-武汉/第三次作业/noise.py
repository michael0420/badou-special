import copy
import random
import cv2 as cv
from skimage import util, img_as_float, io
import numpy as np


class Noise:
    def __init__(self, img_path, prob):
        """
        :param img_path: 图像路径
        :param prob:  噪声比例
        """
        self.image = cv.imread(img_path)
        self.shape = self.image.shape
        self.prob = prob

    def gaussian_noise(self, mean, var):
        """
        高斯噪声
        :param mean: 均值
        :param var:  方差
        :return:
        """
        dst_img = copy.deepcopy(self.image)
        H, W, C = self.shape
        print(H, W, C)
        num = int(H * W * self.prob)
        for _ in range(num):
            rand_x = random.randint(0, H - 1)
            rand_y = random.randint(0, W - 1)

            for c in range(C):
                guass = random.gauss(mean, var)
                dst_img[rand_x][rand_y][c] += guass
        dst_img = np.clip(dst_img, 0, 255)
        cv.imwrite(f"gaussian_noise.png", dst_img)

    def gaussian_noise_sk_image(self):
        image = img_as_float(self.image)
        dst_img = util.random_noise(image, mode="gaussian")
        dst_img.astype('uint8')
        io.imsave(f"gaussian_noise_sk_image.png", dst_img)
        pass

    def salt_pepper_noise(self, thr):
        """
        椒盐噪声
        :param thr: 椒盐阈值 [0,1]
        :return:
        """
        dst_img = copy.deepcopy(self.image)
        H, W, C = self.shape
        print(H, W, C)
        num = int(H * W * self.prob)
        for _ in range(num):
            rand_x = random.randint(0, H - 1)
            rand_y = random.randint(0, W - 1)

            for c in range(C):
                if random.random() <= thr:
                    dst_img[rand_x][rand_y][c] = 0
                else:
                    dst_img[rand_x][rand_y][c] = 255
            # if random.random() <= thr:
            #     dst_img[rand_x][rand_y] = [0, 0, 0]
            # else:
            #     dst_img[rand_x][rand_y] = [255, 255, 255]
        cv.imwrite(f"salt_pepper_noise.png", dst_img)

    def salt_pepper_noise_sk_image(self):
        image = img_as_float(self.image)
        dst_img = util.random_noise(image, mode="s&p")
        dst_img.astype('uint8')
        io.imsave(f"salt_pepper_noise_sk_image.png", dst_img)


if __name__ == '__main__':
    noise = Noise("test.jpeg", 0.8)
    noise.salt_pepper_noise(0.5)
    noise.salt_pepper_noise_sk_image()
    noise.gaussian_noise(2, 4)
    noise.gaussian_noise_sk_image()
