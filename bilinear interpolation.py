import numpy as np
import cv2
def BiL_interpolation(img,out_dim):
    Y_H , Y_W , Y_C = img.shape
    M_H , M_W = out_dim[1] ,out_dim[0]
    if M_H == Y_H | M_W == Y_W:
        return img.copy()
    else:
        ratio_x = float(Y_W/M_W)
        ratio_y = float(Y_H/M_H)
        M_img = np.zeros((M_H,M_W,3),dtype=np.uint8)
        for  i in range(3):
            for m_y in M_H:
                for m_x in M_W:
                    s_x = (m_x + 0.5) * ratio_x - 0.5 # 进行中心重合
                    s_y = (m_y + 0.5) * ratio_y - 0.5

                    s_x0 = int(s_x)
                    s_x1 = min(s_x0 + 1,Y_W - 1)
                    s_y0 = int(s_y)
                    s_y1 = min(s_y0 + 1,Y_H - 1)
                    #实现双线性插值
                    Point0 = (s_x1 - s_x) * img[s_y0,s_x0,i] + (s_x - s_x0) * img[s_y1,s_x0,i]
                    Point1 = (s_x1 - s_x) * img[s_y1,s_x0,i] + (s_x - s_x0 ) * img[s_y1,s_x1,i]
                    M_img[m_y,m_x,i]= (s_y1 - s_y) * Point0 + (s_y - s_y0) * Point1

        return M_img

if __name__ =='__main__':
    img = cv2.imread('lenna.png')
    m_img = BiL_interpolation(img,(800,800))
    cv2.imshow('bilinear interpolation',m_img)
    cv2.waitKey()
    
                    