import cv2
import numpy as np
'''
bilinear interpolation
'''

def bilinear_interpolation(img_src, height_dst, width_dst):
    h_src, w_src, ch_src = img_src.shape
    img_dst = np.zeros([height_dst, width_dst, ch_src], dtype=img_src.dtype)
    ratio_h = h_src / height_dst
    ratio_w = w_src / width_dst
    for i in range(ch_src):
        for j in range(height_dst):
            for k in range(width_dst):
                x_src = (k + 0.5) * ratio_w
                y_src = (j + 0.5) * ratio_h
                x1_src, y1_src = int(np.floor(x_src)), int(np.floor(y_src))
                x2_src, y2_src = min(x1_src + 1, w_src - 1), min(y1_src + 1, h_src - 1)

                # Matrix operation
                x_diff = np.array([x2_src - x_src, x_src - x1_src]).reshape(1,2)
                y_diff = np.array([y2_src - y_src, y_src - y1_src]).reshape(2,1)
                f_Q11 = img_src[y1_src, x1_src, i]
                f_Q12 = img_src[y2_src, x1_src, i]
                f_Q21 = img_src[y1_src, x2_src, i]
                f_Q22 = img_src[y2_src, x2_src, i]
                f_Q = np.array([[f_Q11, f_Q12],[f_Q21, f_Q22]])
                img_dst[j, k, i] = x_diff.dot(f_Q).dot(y_diff).item(0)

                # General operation
                # r1 = (x2_src - x_src) * img_src[y1_src, x1_src, i] + (x_src - x1_src) * img_src[y1_src, x2_src, i]
                # r2 = (x2_src - x_src) * img_src[y2_src, x1_src, i] + (x_src - x1_src) * img_src[y2_src, x2_src, i]
                # img_dst[j, k, i] = np.floor((y2_src - y_src) * r1 + (y_src - y1_src) * r2)

    return img_dst


img_src = cv2.imread("../lenna.png")
img_dst = bilinear_interpolation(img_src, 1024, 1024)
cv2.imshow("Original imagge", img_src)
cv2.imshow("Bilinear interpolation image", img_dst)
cv2.imwrite("Upsampling by the bilinear interpolation.png", img_dst)
cv2.waitKey()