import numpy as np
import cv2

def Zoom_by_Nearest_Interp(img,h,w):
    #
    height,width,channels =img.shape[0:3]
    emptyImage=np.zeros((h,w,channels),np.uint8)
    scale_of_h=h/height
    scale_of_w=w/width
    for i in range(h-1):
        for j in range(w-1):
            x=int(0.5 + i/scale_of_h )
            y=int(0.5 + j/scale_of_w )
            emptyImage[i,j]=img[x,y]
    return emptyImage
def Zoom_by_Bilinear_Interp(img,zoom_width,zoom_height):
    #
    height, width, channels = img.shape[0:3]
    zoom_img = np.zeros((zoom_height,zoom_width, 3), dtype=np.uint8)
    scale_w, scale_h = float(width) / zoom_width, float(height) / zoom_height
    for i in range(3):
        for dst_y in range(zoom_height):
            for dst_x in range(zoom_width):
                src_x = int((dst_x + 0.5) * scale_w - 0.5)
                src_y = int((dst_y + 0.5) * scale_h - 0.5)

                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_x - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_y - 1)

                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                zoom_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return zoom_img

def Equalize_Hist_Color(img):
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    result = cv2.merge((bH, gH, rH))
    return result



img = cv2.imread("C:\\openCV\\opencv\\sources\\modules\\core\\misc\\objc\\test\\resources\\lena.png",1)
img1 = cv2.imread("mydrawing.jpg",1)
#img = cv2.imread("lena.png",1)
zoom_1=Zoom_by_Nearest_Interp(img,750,500)
zoom_2=Zoom_by_Bilinear_Interp(img,1000,500)
equalized_img=Equalize_Hist_Color(img1)
print(img.shape)
print(zoom_1.shape)
print(zoom_2.shape)
cv2.imshow("Zoom_by_nearestinterp_image", zoom_1)
cv2.imshow("Zoom_by_bilinearinterp_image", zoom_1)
cv2.imshow("origin_image", img)
cv2.imshow("image_after_histequalization", equalized_img)
cv2.imshow("image_before_histequalization", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()