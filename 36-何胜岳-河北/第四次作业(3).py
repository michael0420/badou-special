import numpy as np
import cv2

img=cv2.imread('photo1.jpg')
F_Z=img.copy()

scr=np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst=np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
print(img.shape)

#求warpMatrix矩阵
m=cv2.getPerspectiveTransform(scr,dst)


print('warpMatrix')
print(m)

#利用warpMatrix矩阵进行图像变换
result=cv2.warpPerspective(F_Z,m,(337,488))
## cv2.warpPerspective(src，M，dsize，dst，flags，borderMode，borderValue)
# src：输入图像
# M：变换矩阵
# dsize：目标图像shape
# flags：插值方式，interpolation方法INTER_LINEAR或INTER_NEAREST
# borderMode：
# 边界补偿方式，BORDER_CONSTANTorBORDER_REPLICATE
# borderValue：边界补偿大小，常值，默认为0

# warpPerspective适用于图像。perspectiveTransform适用于一组点。

cv2.imshow('img',img)
cv2.imshow('result',result)
cv2.imwrite('photo2.jpg',result)
cv2.waitKey(0)












