import cv2
import numpy as np

#

# Nearest interp 最邻近插值
# 求空白与原图的比例，然后空白每个坐标值对应的原图坐标值（用对应画布比例可求）->再赋值原图坐标图像值
def Nearest_interp (img):
    height,width,channels = img.shape
    emptyImg = np.zeros((1000,1000,channels),np.uint8)
    sh=1000/height
    sw=1000/width
    for i in range(1000):
        for j in range(1000):
            x=int(i/sh+0.5)         #求对应原图坐标值（整数），+0.5实现四舍五入
            y=int(j/sw+0.5)
            emptyImg[i,j]=img[x,y]
    return emptyImg

# path= "lenna.png"
img=cv2.imread("lenna.png")
zoomImg=Nearest_interp(img)
print(zoomImg)
print(zoomImg.shape)
cv2.imshow("nearest interp",zoomImg)
cv2.imshow("image",img)
cv2.waitKey()


#bilinear interpolation 双线性插值
#运用kx+b思想求P点的坐标值，均用目标图像坐标求原图坐标
def bilinear_interpolation(img,out_dim):
    src_h,src_w,channels = img.shape
    #0行1列
    dst_h,dst_w = out_dim[0], out_dim[1]
    print("src_h, src_w=", src_h, src_w)
    print("dst_h, dst_w=", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h,dst_w,channels),np.uint8)
    scale_x, scale_y = float(src_w/dst_w), float(src_h/dst_h)
    for i in range(channels):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):

                #中心点重合情况下, 找对应原图坐标
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                #锁定用于计算目标坐标的四个原图点
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1 , src_w - 1)    #避免超出画布宽
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1 , src_h-1)

                #计算R1，R2，再用R1，R2算P点图像值
                #矩阵行是heigh-此为y，列是width-此为x
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img

if __name__ == '__main__':
    img=cv2.imread('lenna.png')
    dst=bilinear_interpolation(img,(1000,700))
    cv2.imshow('bilinear interp', dst)
    cv2.waitKey()