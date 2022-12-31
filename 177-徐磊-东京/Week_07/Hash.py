"""
@author: Rai
均值和差值hash算法
"""

import cv2
import numpy as np

def ave_hash(img):
    img = cv2.resize(img, (8,  8), interpolation=cv2.INTER_CUBIC)  # 三次样条差值
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ave = img.mean()
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if img[i, j] > ave:
                hash_str += '1'
            else:
                hash_str += '0'
    return hash_str


def diff_hash(img):
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if img[i,j] > img[i, j+1]:
                hash_str += '1'
            else:
                hash_str += '0'
    return hash_str


def cmp_hash(hash1, hash2):
    n1 = len(hash1)
    n2 = len(hash2)
    if n1 != n2:
        return -1
    cnt = 0
    for i in range(n1):
        if hash1[i] != hash2[i]:
            cnt += 1
    return cnt


img1 = cv2.imread('lenna.png')
img2 = cv2.imread('lenna_noise.png')
hash1 = ave_hash(img1)
hash2 = ave_hash(img2)
print('---均值哈希算法的哈希值:---')
print('hash1:', hash1)
print('hash2:', hash2)

n = cmp_hash(hash1, hash2)
if n == -1:
    print('传入的参数不对')
else:
    print('均值哈希算法相似度:', n)


hash1 = diff_hash(img1)
hash2 = diff_hash(img2)
print('\n---差值哈希算法的哈希值---')
print('hash1:', hash1)
print('hash2:', hash2)
n = cmp_hash(hash1, hash2)
if n == -1:
    print('传入的参数不对')
else:
    print('差值哈希算法相似度:', n)