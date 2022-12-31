import numpy as np
import cv2 as cv


class TransformTool:
    def __init__(self, img_path):
        self.image = cv.imread(img_path)
        self.shape = self.image.shape
        self.warp_matrix = None
        self.warp_matrix_I = None
        pass

    def computer_warp_matrix(self, src, dst):
        """
        :param src: 原图坐标
        :param dst: 目标图坐标
        :return:
        """
        """
                         [[w11,w12,w13],
        [X,Y,Z] = [u,v,1] [w21,w22,w23],
                          [w31,w32, 1 ]]
        
        x = X/Z = (w11*u + w21*v + w31)/(w13*u + w23*v +1)
        y = Y/Z = (w12*u + w22*v + w32)/(w13*u + w23*v +1)
        
        (w13*u + w23*v +1)*x = w11*u + w21*v + w31
        (w13*u + w23*v +1)*y = w12*u + w22*v + w32
        
        x = w11*u + w21*v + w31 - u*x*w13 - v*x*w23  
        y = w12*u + w22*v + w32 - u*y*w13 - v*y*w23
        
        """
        # A 8*8  B 8*1
        A = np.zeros((8, 8))
        B = np.zeros((8, 1))
        for i in range(4):
            A[2 * i, :] = [src[i][0], 0, -src[i][0] * dst[i][0], src[i][1], 0, -src[i][1] * dst[i][0], 1, 0]
            A[2 * i + 1, :] = [0, src[i][0], -src[i][0] * dst[i][1], 0, src[i][1], -src[i][1] * dst[i][1], 0, 1]
            B[2 * i] = dst[i][0]
            B[2 * i + 1] = dst[i][1]

        # 计算 A 矩阵的逆
        A = np.mat(A)
        self.warp_matrix = A.I * B

        self.warp_matrix = np.array(self.warp_matrix).T[0]
        self.warp_matrix = np.insert(self.warp_matrix, self.warp_matrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
        self.warp_matrix = self.warp_matrix.reshape((3, 3))

        self.warp_matrix = np.mat(self.warp_matrix)
        self.warp_matrix_I = self.warp_matrix.I

    def WarpPerspectiveMatrix(self, src, dst):
        src = np.float32(src)
        dst = np.float32(dst)
        assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4

        nums = src.shape[0]
        A = np.zeros((2 * nums, 8))  # A*warpMatrix=B
        B = np.zeros((2 * nums, 1))
        for i in range(0, nums):
            A_i = src[i, :]
            B_i = dst[i, :]
            A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0,
                           -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
            B[2 * i] = B_i[0]

            A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1,
                               -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
            B[2 * i + 1] = B_i[1]

        A = np.mat(A)
        # 用A.I求出A的逆矩阵，然后与B相乘，求出warpMatrix
        warpMatrix = A.I * B  # 求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32

        # 之后为结果的后处理
        warpMatrix = np.array(warpMatrix).T[0]
        warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
        warpMatrix = warpMatrix.reshape((3, 3))
        print("warpMatrix:\n")
        print(warpMatrix)
        return warpMatrix

    def transform(self):

        dst_img = cv.warpPerspective(self.image, self.warp_matrix.T, (761, 529))
        cv.imwrite(f"transform.png", dst_img)

    def transform_cv2(self, src, dst):
        src = np.float32(src)
        dst = np.float32(dst)
        self.warp_matrix = cv.getPerspectiveTransform(src, dst)


if __name__ == '__main__':
    src = [[200, 0], [380, 0], [0, 529], [290, 529]]
    dst = [[0, 0], [290, 0], [0, 529], [290, 529]]
    transform_tool = TransformTool("test_v1.jpeg")

    transform_tool.computer_warp_matrix(src, dst)

    transform_tool.transform()
