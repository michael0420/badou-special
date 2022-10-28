import cv2
import numpy as np

def zoom(img, h, w):
    height, width, channels = img.shape
    blank = np.zeros((h, w, channels),  dtype='uint8')
    sh = h / height
    sw = w / width
    for j in range(h):
        for i in range(w):
            y = int(j / sh + 0.5)
            x = int(i / sw + 0.5)
            blank[j, i] = img[y, x]
    return blank

if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    print(img[20])
    cv2.imshow("image nearest", zoom(img, 700, 600))
    cv2.imshow("image origin", img)
    cv2.waitKey(0)
