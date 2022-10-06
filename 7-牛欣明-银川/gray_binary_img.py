# author:牛欣明
# createTime: 2022年10月6日18:08:17
# describe:transform rgb img to gray 将rgb-img转换为灰色
import cv2
import numpy as np

if __name__ == '__main__':
    img = cv2.imread("pic/lenna.png")
    # 获取图片高河宽
    h, w = img.shape[:2]
    target_img = np.zeros((h, w), np.float32)
    for i in range(h):
        for j in range(w):
            target_img[i, j] = int(sum([img[i][j][0], img[i][j][1], img[i][j][2]]) / 3)
    gray_img = target_img.astype(np.uint8)
    binary_img = np.where(target_img <= 150, 0, 255).astype(np.uint8)

    cv2.imshow("src_img",img)
    cv2.imshow("gray_img",gray_img)
    cv2.imshow("binary_img",binary_img)
    cv2.waitKey()
