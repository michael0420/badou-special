import cv2 as cv

def rgb2gray(use_cv=False):
    if use_cv:
        # img = cv.imread(r'week2\lenna.png', cv.IMREAD_GRAYSCALE)
        ori = cv.imread(r'week2\lenna.png')
        img = cv.cvtColor(ori, cv.COLOR_BGR2GRAY)
    else:
        ori = cv.imread(r'week2\lenna.png')
        img = (ori[:,:,0] * 0.11 + ori[:,:,1] * 0.59 + ori[:,:,2] * 0.3) / 255
        
    cv.imshow("lenna", img)
    cv.waitKey()

if __name__ == "__main__":
    use_cv=False
    rgb2gray(use_cv)