import cv2
import numpy as np


def bilinear_interpolation(img, out_dim):
    src_h, src_w, channel = img.shape
    tar_h, tar_w = out_dim[1], out_dim[0]
    print("src_h, src_w = ", src_h, src_w)
    print("tar_h, tar_w = ", tar_h, tar_w)
    if src_h == tar_h and src_w == tar_w:
        return img.copy()
    tar_img = np.zeros((tar_h, tar_w, channel), dtype=np.uint8)
    # 计算放缩比例
    scale_x, scale_y = float(src_w) / tar_w, float(src_h) / tar_h
    for i in range(channel):
        for tar_y in range(tar_h):
            for tar_x in range(tar_w):
                # 找到原图中对应的像素位置（有可能不存在）
                src_x = (tar_x + 0.5) * scale_x - 0.5
                src_y = (tar_y + 0.5) * scale_y - 0.5

                # 框出原图中需要插值对应的点（实际存在）
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # 计算要插入的值
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                tar_img[tar_y, tar_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return tar_img

if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    tar = bilinear_interpolation(img, (800, 800))
    cv2.imshow('interp_bilinear', tar)
    cv2.waitKey()