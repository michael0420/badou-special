"""
步骤
1. 缩放：图片缩放为8*9，保留结构，除去细节。
2. 灰度化：转换为灰度图。
3. 求平均值：计算灰度图所有像素的平均值。 ---这步没有，只是为了与均值哈希做对比
4. 比较：像素值大于后一个像素值记作1，相反记作0。本行不与下一行对比，每行9个像素，
八个差值，有8行，总共64位
5. 生成hash：将上述步骤生成的1和0按顺序组合起来既是图片的指纹（hash）。
6. 对比指纹：将两幅图的指纹对比，计算汉明距离，即两个64位的hash值有多少位是不一样的，不相同位数越少，图片越相似。
"""
import cv2


def read_image(path):
    return cv2.imread(path)


def resize_image(path, resize):
    image = read_image(path)
    img = cv2.resize(image, resize, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def mean_hash(path, resize=(9, 8)):
    gray = resize_image(path, resize)
    s = ''
    for i in range(resize[1]):
        for j in range(resize[0] - 1):
            if gray[i, j] > gray[i, j + 1]:
                s += '1'
            else:
                s += '0'
    return s


def compare_spread_hash(hash1, hash2):
    n = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] == hash2[i]:
            n += 1
    return n, round(n / len(hash1), 2)


if __name__ == '__main__':
    right, rate = compare_spread_hash(mean_hash("./data/lenna.png"), mean_hash("./data/lenna_noise.png"))
    print(f"相同的个数有:{right},相似度：{rate}")
