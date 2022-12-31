import cv2 as cv
import numpy as np

def aHash(img1, img2):
    def _get_code(img):
        code = ''
        pix_mean = img.mean()
        img = img.flatten()
        for each in img:
            if each > pix_mean:
                code += '1'
            else:
                code += '0'
        return code
        
    img1 = cv.resize(img1, (8, 8), interpolation=cv.INTER_CUBIC)
    img2 = cv.resize(img2, (8, 8), interpolation=cv.INTER_CUBIC)
    
    code1, code2 = _get_code(img1), _get_code(img2)
    print(code1)
    print(code2)
    comp_hash_code(code1, code2)

def dHash(img1, img2):
    def _get_code(img):
        code = ''
        for i in range(img.shape[0]):
            for j in range(img.shape[1] - 1):
                if img[i, j] > img[i, j+1]:
                    code += '1'
                else:
                    code += '0'
        return code
        
    img1 = cv.resize(img1, (9, 8), interpolation=cv.INTER_CUBIC)
    img2 = cv.resize(img2, (9, 8), interpolation=cv.INTER_CUBIC)
    
    code1, code2 = _get_code(img1), _get_code(img2)
    print(code1)
    print(code2)
    comp_hash_code(code1, code2)

def comp_hash_code(code1, code2):
    n = 0
    for i in range(len(code1)):
        if code1[i] != code2[i]:
            n += 1
    print(f'不一样的个数为：{n}, 相似度为：{1 - n / len(code1)}')

if __name__ == "__main__":
    img1 = cv.imread(r'week7\lenna.png', cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(r'week7\lenna_noise.png', cv.IMREAD_GRAYSCALE)
    aHash(img1, img2)
    dHash(img1, img2)