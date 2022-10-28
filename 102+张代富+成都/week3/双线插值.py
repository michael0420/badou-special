import cv2
import numpy as np


def bilinear_interpolation(source_image, shape):
    src_h, src_w, channel = source_image.shape  # 原图片的高、宽、通道数
    dst_h, dst_w = shape[1], shape[0]  # 输出图片的高、宽
    if src_h == dst_h and src_w == dst_w:
        return source_image
    dst_image = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 源图像和目标图像几何中心的对齐
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # 计算在源图上四个近邻点的位置
                src_x0 = int(np.floor(src_x))
                src_y0 = int(np.floor(src_y))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y1 = min(src_y0 + 1, src_h - 1)

                # 双线性插值
                temp0 = (src_x1 - src_x) * source_image[src_y0, src_x0, i] + (src_x - src_x0) * source_image[
                    src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * source_image[src_y1, src_x0, i] + (src_x - src_x0) * source_image[
                    src_y1, src_x1, i]
                dst_image[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_image


if __name__ == '__main__':
    # h,w,c
    image = cv2.imread("../data/lenna.png")
    image2 = bilinear_interpolation(image, (1000, 800))
    cv2.imshow('image', image)
    cv2.imshow('image2', image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
