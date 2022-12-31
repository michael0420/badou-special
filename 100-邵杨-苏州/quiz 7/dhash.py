import cv2 as cv
import numpy as np

def dhash(img, width = 9, height = 8):
    img_list = []
    img = cv.resize(img, (9, 8), interpolation=cv.INTER_LINEAR)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    for j in range(img.shape[0]):
        for i in range(img.shape[1]-1):
            if img[j, i] > img[j, i+1]:
                img_list.append(1)
            else:
                img_list.append(0)
    return img_list

def compare_hash(hash1, hash2):
    count = int(0)
    if len(hash1) != len(hash2):
        print('error')
    else:
        for i in range(len(hash1)):
            if hash1[i] != hash2[i]:
                count += 1
    return count

img1 = cv.imread('lenna.png')
img2 = cv.imread('lenna_noise.png')
hash1 = dhash(img1)
hash2 = dhash(img2)
print(hash1)
print(hash2)
print('差值哈希算法相似度：', compare_hash(hash1, hash2))