import numpy as np
import cv2


def WarpMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4

    nums = src.shape[0]
    A = np.zeros((2 * nums, 8))
    B = np.zeros((2 * nums, 1))
    # 透视变换
    for i in range(0, nums):
        A_i = src[i, :]
        B_i = dst[i, :]
        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[2 * i] = B_i[0]
        A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i + 1] = B_i[1]

    A = np.mat(A)
    warpmatrix = A.I * B

    warpmatrix = np.array(warpmatrix).T[0]
    warpmatrix = np.insert(warpmatrix, warpmatrix.shape[0], values=1.0, axis=0)
    warpmatrix = warpmatrix.reshape((3, 3))
    return warpmatrix


if __name__ == '__main__':
    img = cv2.imread('photo1.jpg')
    src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

    m = WarpMatrix(src, dst)
    print("warpmatrix", m)
    result = cv2.warpPerspective(img, m, (337, 488))

    cv2.imshow('img', img)
    cv2.imshow('result', result)
    cv2.waitKey()



