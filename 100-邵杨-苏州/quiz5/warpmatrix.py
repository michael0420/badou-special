import numpy as np
import cv2 as cv

def warptransformation(src, dst):  # A*warpMatrix=B
    number = src.shape[0]
    A = np.ones((2 * number, 8))
    B = np.ones((2 * number, 1))
    for i in range(number):
        A_i = src[i, :]
        B_i = dst[i, :]
        A[2*i, :] = [A_i[0],  A_i[1],  1,
                     0,      0,       0,
                     -A_i[0]*B_i[1], -A_i[1]*B_i[0]]
        A[2*i+1, :] = [0,      0,     0,
                       A_i[0],  A_i[1],  1,
                       -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2*i] = B_i[0]
        B[2*i + 1] = B_i[1]

    A = np.mat(A)  #array 转换为矩阵
    matrix = np.dot(A.I, B)
    matrix = matrix.tolist() #矩阵变为列表
    matrix.append([1.0])
    matrix = np.array(matrix).reshape([3, 3])
    return matrix


if __name__ == '__main__':
    print('matrix')
    src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
    Matrix = warptransformation(src, dst)
    print(Matrix)

    img = cv.imread('photo1.jpg')
    result = cv.warpPerspective(img, Matrix, (337, 488))
    cv.imshow("origin", img)
    cv.imshow("result", result)
    cv.waitKey(0)

