import cv2
import numpy as np


def nearest_insert(old_image, w2, h2):
    h, w, c = old_image.shape
    new_image = np.zeros((h2, w2, c), np.uint8)
    sw = w2 / w
    sh = h2 / h
    for i in range(h2):
        for j in range(w2):
            x = int(i / sh)
            y = int(j / sw)
            new_image[i, j] = old_image[x, y]
    return new_image


if __name__ == '__main__':
    # h,w,c
    image = cv2.imread("../data/lenna.png")
    print(image.shape)
    image2 = nearest_insert(image, 1024, 1000)
    print(image2.shape)
    cv2.imshow('image', image)
    cv2.imshow('image2', image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
