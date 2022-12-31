import cv2 as cv
import numpy as np

def bilinear (img, o_dim):
    sh, sw, sc = img.shape
    print('sh,sw=', sh, sw)
    oh, ow = o_dim[1], o_dim[0]
    if sh == oh and sw == ow:
        return img
    out_img = np.zeros((oh, ow, 3), dtype=np.uint8)
    rx, ry = float(sw)/ow, float(sh)/oh
    # 找原图对应坐标并对齐中心
    for c in range(3):
        for oy in range(oh):
            for ox in range(ow):
                sx = (ox - 0.5)*rx - 0.5
                sy = (oy - 0.5)*ry - 0.5
            # 求原图四个点坐标
                x1 = int(np.floor(sx))
                x2 = min(x1+1, sw - 1)
                y1 = int(np.floor(sy))
                y2 = min(y1 + 1, sh - 1)

            # 对x轴，y轴进行插值，对结果再进行一次插值;对于图像数据，img.[0]是H,[1]是w
                r1 = (x2-sx)*img[y1, x1, c]+(sx-x1)*img[y1, x2, c]
                r2 = (x2-sx)*img[y2, x1, c]+(sx-x1)*img[y2, x2, c]

                out_img[oy, ox, c] = int((y2-sy)*r1 + (sy-y1)*r2)

    return out_img

if __name__ == "__main__":
    img = cv.imread('lenna.png')
    dst = bilinear(img, (800, 800))
    cv.imshow('image',img)
    cv.imshow('bilinear_img',dst)
    cv.waitKey(0)


