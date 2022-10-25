import os,sys
import cv2
import numpy as np

def rgb2gray(im):
    #Gray = R*0.299 + G*0.587 + B*0.114
    if im.shape[2] != 3:
        print("channel num is error")
        return []
    new_im = im[:,:,0]*0.299+im[:,:,1]*0.587+im[:,:,2]*0.114
    #new_im = new_im/1000.0
    new_im = np.clip(new_im,0,255)
    return new_im

def gray2bin(im,bin_thre):
    if im.shape[0] == 0:
        print("gray im is empty")
        return []
    new_im = np.zeros((im.shape[0],im.shape[1]))
    mask = im > bin_thre
    new_im[mask] += 255
    return new_im

def main(img_path,opt):
    opt = int(opt)
    if not os.path.exists(img_path):
        print("%s not exists" % img_path)
        return
    im = cv2.imread(img_path)
    if opt == 0:
        gray_im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        cv2.imwrite("./gray.jpg",gray_im)
        _,bin_im = cv2.threshold(gray_im,100,255,cv2.THRESH_BINARY)
        cv2.imwrite("./bin.jpg",bin_im)
    else:
        rgb_im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        gray_im = rgb2gray(rgb_im)
        cv2.imwrite("./gray.jpg",gray_im)
        bin_im = gray2bin(gray_im,100)
        cv2.imwrite("bin.jpg",bin_im)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("python work1.py img_path 0|1 0:lib 1:self")
    else:
        main(sys.argv[1],sys.argv[2])

