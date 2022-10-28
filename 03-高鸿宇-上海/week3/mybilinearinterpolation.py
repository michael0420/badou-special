import numpy as np
import cv2 as cv

def my_bilinear_interpolation(size, img):
    dst_width, dst_height = size
    ori_width, ori_height, channels = img.shape
    if (size == img.shape[:2]):
        return img.copy()
    dst = np.zeros((dst_width, dst_height, channels), dtype=img.dtype)
    w_scale = ori_width / dst_width
    h_scale = ori_height / dst_height
    
    for c in range(channels):
        for i in range(dst_height):
            for j in range(dst_width):
                x = (i + 0.5) * h_scale - 0.5
                y = (j + 0.5) * w_scale - 0.5
                
                x0 = int(x)
                x1 = min(x0+1, ori_height-1)
                y0 = int(y)
                y1 = min(y0+1, ori_width-1)
                
                r0 = (y1 - y) * img[x0, y0, c] + (y - y0) * img[x0, y1, c]
                r1 = (y1 - y) * img[x1, y0, c] + (y - y0) * img[x1, y1, c]
                
                dst[i, j, c] = int((x1 - x) * r0) + int((x - x0) * r1)
    
    return dst

if __name__ == "__main__":
   ori = cv.imread(r'week2\lenna.png')
   dst_size = (800, 800)
   dst = my_bilinear_interpolation(dst_size, ori)
   cv.imshow("lenna", dst)
   cv.waitKey(0)