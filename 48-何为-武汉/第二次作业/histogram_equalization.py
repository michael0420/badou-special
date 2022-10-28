import cv2 as cv
import numpy as np
import math
import copy


class HistogramEqualization:
    """
    """

    def __init__(self, img_path):
        self.image = cv.imread(img_path)
        self.shape = self.image.shape

    def histogram_equalization(self):
        dst_img = copy.deepcopy(self.image)
        hist_acc = self.histogram_statistics()
        H, W, C = self.shape
        for c in range(C):
            for i in range(H):
                for j in range(W):
                    dst_img[i][j][c] = hist_acc[self.image[i][j][c]][c] * 256 / (H * W) - 1
        cv.imwrite(f"histogram_equalization.png", dst_img)

    def histogram_statistics(self):
        """
        图像统计
        :return:
        """
        # 直方图统计对象是 单个通道
        # 多通道对每个通道 分别进行处理
        hist = np.zeros(shape=(256, 3))
        hist_acc = np.zeros(shape=(256, 3))
        H, W, C = self.shape
        # for c in range(C):
        for i in range(H):
            for j in range(W):
                for c in range(C):
                    hist[self.image[i][j][c]][c] += 1

        # for c in range(C):
        hist_acc[0] = hist[0]
        for i in range(1, 256):
            hist_acc[i] = hist[i] + hist_acc[i - 1]

        return hist_acc

    def histogram_equalization_cv(self):

        chans = cv.split(self.image)
        # 直方图
        hist = cv.calcHist([chans[0]], [0], None, [256], [0, 256])
        bh = cv.equalizeHist(chans[0])
        gh = cv.equalizeHist(chans[1])
        rh = cv.equalizeHist(chans[2])
        dst_img = cv.merge((bh, gh, rh))

        cv.imwrite(f"histogram_equalization_CV.png", dst_img)


if __name__ == '__main__':
    #
    histogram = HistogramEqualization("test.jpeg")
    histogram.histogram_equalization()
