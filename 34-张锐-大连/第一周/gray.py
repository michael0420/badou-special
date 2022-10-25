import cv2 as cv2
import numpy as np
import timeit


def RGB2Gray(img):
    img = cv2.imread(img)
    cv2.imshow("origin", img)
    key = cv2.waitKey(0)
    h, w = img.shape[:2]
    img_gray = np.zeros([h, w], img.dtype)
    for i in range(h):
        for j in range(w):
            m = img[i, j]
            img_gray[i, j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
    #print(img)
    #print("----: %s" % img_gray)
    cv2.imshow("RGB2Gray", img_gray)
    key = cv2.waitKey(0)

def threshold(img, thresh):
    img = cv2.imread(img)
    cv2.imshow("origin", img)
    key = cv2.waitKey(0)
    h, w = img.shape[:2]
    img_gray = np.zeros([h, w], img.dtype)
    for i in range(h):
        for j in range(w):
            m = img[i, j]
            img_gray[i, j] = 0 if int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3) < thresh else 255
    #print(img)
    #print("----: %s" % img_gray)
    cv2.imshow("threshold", img_gray)
    key = cv2.waitKey(0)


#RGB2Gray('1.png')
#threshold('1.png', 200)

img = cv2.imread('1.png')
uimg = cv2.UMat(img)
#t_BGR2GRAY = timeit.timeit("img_test(img, 'img')", globals=globals(), number=10000)
t_BGR2GRAY = timeit.timeit("cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)", globals=globals(), number=10000)
print("time for cpu BGR2GRAY: ", t_BGR2GRAY)
#t_UBGR2GRAY = timeit.timeit("img_test(img, 'UMat')", globals=globals(), number=10000)
t_UBGR2GRAY = timeit.timeit("cv2.cvtColor(uimg, cv2.COLOR_BGR2GRAY)", globals=globals(), number=10000)
print("time for gpu BGR2GRAY: ", t_UBGR2GRAY)



