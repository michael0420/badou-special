import numpy as np
import cv2 as cv

def my_Nearest_Neighbor_Interpolation(size, img):
    dst_width, dst_height = size
    ori_width, ori_height, channels = img.shape
    dst = np.zeros((dst_width, dst_height, channels), dtype=img.dtype)
    w_scale = ori_width / dst_width
    h_scale = ori_height / dst_height
    
    for i in range(dst_height):
        for j in range(dst_width):
            x = int(i * h_scale + 0.5)
            y = int(j * w_scale + 0.5)
            dst[i, j, :] = img[x, y, :]
    
    return dst

if __name__ == "__main__":
   ori = cv.imread(r'week2\lenna.png')
   dst_size = (800, 800)
   dst = my_Nearest_Neighbor_Interpolation(dst_size, ori)
   cv.imshow("lenna", dst)
   cv.waitKey(0)