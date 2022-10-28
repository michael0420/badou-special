import cv2 as cv
import numpy as np

def bilinear_interpolation(img, result_img):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = result_img[1], result_img[0]
    dst_img = np.zeros((dst_h, dst_w, 3),  dtype='uint8')
    scale_x, scale_y = src_w / dst_w, src_h / dst_h
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                scr_x = (dst_x + 0.5) * scale_x - 0.5
                scr_y = (dst_y + 0.5) * scale_y - 0.5

                scr_x0 = int(scr_x)
                scr_y0 = int(scr_y)
                scr_x1 = min(scr_x0 + 1, src_w - 1)
                scr_y1 = min(scr_y0 + 1, src_h - 1)

                tem0 = (scr_x1 - scr_x) * img[scr_y0, scr_x0, i] + (scr_x - scr_x0) * img[scr_y0, scr_x1, i]
                tem1 = (scr_x1 - scr_x) * img[scr_y1, scr_x0, i] + (scr_x - scr_x0) * img[scr_y1, scr_x1, i]
                dst_img[dst_y, dst_x, i] = int((scr_y1 - scr_y) * tem0 + (scr_y - scr_y0) * tem1)
    return dst_img

if __name__ == '__main__':
    img = cv.imread('lenna.png')
    dst = bilinear_interpolation(img, (700, 700))
    cv.imshow('origin', img)
    cv.imshow('bilinear', dst)
    cv.waitKey(0)
