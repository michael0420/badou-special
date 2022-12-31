import cv2 as cv
import numpy as np

def my_perspective_transform(src, src_points, dst_points, size, use_cvfun=True):
    if use_cvfun:
        wrap_martrix = cv.getPerspectiveTransform(src_points, dst_points)
        print('wrap_martrix')
        print(wrap_martrix)
        dst = cv.warpPerspective(src, wrap_martrix, size)
    else:
        A = np.zeros(shape=(8, 8))
        for i in range(0, len(A), 2):
            A[i, 0] = src_points[int(i/2), 0]
            A[i, 1] = src_points[int(i/2), 1]
            A[i, 2] = 1
            A[i, 6] = -src_points[int(i/2), 0] * dst_points[int(i/2), 0]
            A[i, 7] = -src_points[int(i/2), 1] * dst_points[int(i/2), 0]
            
            A[i+1, 3] = src_points[int(i/2), 0]
            A[i+1, 4] = src_points[int(i/2), 1]
            A[i+1, 5] = 1
            A[i+1, 6] = -src_points[int(i/2), 0] * dst_points[int(i/2), 1]
            A[i+1, 7] = -src_points[int(i/2), 1] * dst_points[int(i/2), 1]
        dst_matrix = dst_points.flatten()
        a = np.ones((9, ))
        a[:8] = np.linalg.inv(A) @ dst_matrix
        wrap_martrix = a.reshape(3, 3)
        print('wrap_martrix')
        print(wrap_martrix)
        dst = cv.warpPerspective(src, wrap_martrix, size)
    return dst

if __name__ == '__main__':
    img = cv.imread(r'week5\photo1.jpg')
    src_points = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    dst_points = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
    size = (337, 488)
    dst_withmodel = my_perspective_transform(img, src_points, dst_points, size)
    dst = my_perspective_transform(img, src_points, dst_points, size, use_cvfun=False)
    cv.imshow('paper_src', img)
    cv.imshow("paper", np.concatenate((dst_withmodel, dst), 1))
    cv.waitKey(0)