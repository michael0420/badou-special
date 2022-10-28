# #==========================    2:实现双线性插值    ======================

import numpy as np
import cv2

def bilinear_interpolation(img, out_dim):      # outdim是输出（即目标图）的shape（H，W）
    src_h, src_w, channel = img.shape          # 原图像的高，宽以及通道数
    dst_h, dst_w = out_dim[1], out_dim[0]      # 输出图像的高，宽
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    # if src_h == dst_h and src_w == dst_w:
        # return img.copy()
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    # 建立新图像：高和宽全为0，通道数为3的矩阵，并进行保存

    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    # 原图像的宽高和新图像宽高的一个比例关系（有小数值，则用float进行保存）

    for i in range(3):  # 用i对通道数进行循环（指定通道数对channel循环）
        for dst_y in range(dst_h):  # 高度在新图像的高进行循环
            for dst_x in range(dst_w):  # 宽度在新图像的宽进行循环

                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5
                # 原图像的中心和目标新图像的中心能够重合

                src_x0 = int(np.floor(src_x))
                # np.floor()返回不大于输入参数的最大整数。（向下取整）

                src_x1 = min(src_x0 + 1, src_w - 1)   #边界值处理
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)
                # 计算在原图上四个近邻点的位置

                # calculate the interpolation
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
                # 双线性插值（双线性插值公式的代码实现）
                # retimg[i,j]=(1-u)*(1-v)*img[x,y] + u*(1-v)*img[x+1,y] + (1-u)*v*img[x,y+1] + u*v*img[x+1,y+1]
                # retimg[i,j]=(1-u)*(1-v)*img[x,y] + u*(1-v)*img[x+1,y] + (1-u)*v*img[x,y+1] + u*v*img[x+1,y+1]


    return dst_img


if __name__ == '__main__':
    img = cv2.imread('lenna.png')                     #导入图像
    dst = bilinear_interpolation(img, (700, 700))
    cv2.imshow('bilinear interp', dst)                #双线性插值的图像显示
    cv2.waitKey()                                     #保持图像的长时间显示
