import cv2
import numpy as np
img=cv2.imread("photo1.jpg")
old=np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
new=np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
bianhuanjuzhen=cv2.getPerspectiveTransform(old,new)
new_img=cv2.warpPerspective(img,bianhuanjuzhen,(337,488))
cv2.imshow('old',img)
cv2.imshow('new_img',new_img)
cv2.waitKey()