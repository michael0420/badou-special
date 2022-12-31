"""
@author:Rai
透视变换
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2


def getPerspectiveMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4

    n = src.shape[0]
    A = np.zeros((2 * n, 8))
    B = np.zeros((2 * n, 1))
    for i in range(n):
        A_item = src[i, :]
        B_item = dst[i, :]
        A[2 * i, :] = [A_item[0], A_item[1], 1, 0, 0, 0,
                       -A_item[0] * B_item[0], -A_item[1] * B_item[0]]
        A[2 * i + 1, :] = [0, 0, 0, A_item[0], A_item[1], 1,
                           -A_item[0] * B_item[1], -A_item[1] * B_item[1]]
        B[2 * i] = B_item[0]
        B[2 * i + 1] = B_item[1]

    A = np.mat(A)
    warpMatrix = A.I * B

    warpMatrix = np.array(warpMatrix).T[0]  # (1,8) → (8)
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape, values=1.0)
    warpMatrix = warpMatrix.reshape(3, 3)

    return warpMatrix


if __name__ == '__main__':
    src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    # dst = np.float32([[17, 151], [354, 151], [17, 639], [354, 639]])
    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

    warpMatrix = getPerspectiveMatrix(src, dst)
    print('Perspective Transform Matrix :\n', warpMatrix)

    img = cv2.imread('PerspectiveTransform_src.jpg')
    h, w, _ = img.shape
    img_dst = cv2.warpPerspective(img, warpMatrix, (337, 488))
    cv2.imshow('Perspective Transform image', img_dst)
    cv2.imwrite('PerspectiveTransofrm_dst.png', img_dst)
    cv2.waitKey(0)

    # img = plt.imread('PerspectiveTransform_src.jpg')
    # plt.imshow(img)
    # plt.show()
