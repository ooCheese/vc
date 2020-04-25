'''
@author: Lukas Wedemeyer
'''

import cv2
import numpy as np

LIGHT_UP_FACTOR = 0.6

def remap(img,min,max):
    '''s
    mapping the min and max pixel value to min , max
    '''

    factor = (max - min) / (img.max()-img.min())
    return (img * factor + min).astype(np.uint8)

def light_up_img(img,factor):
    '''
    ligth up a image with a factor
    img = (source) Image 
    factor = ligth up factor
    '''
    return remap(img,255*factor,255)


def calc_pix_luminance(left,right,top):
    '''
    calculate luminace for a single pixel
    left = greyValue (left Img)
    right = greyValue (right Img)
    top = greyValue (top Img)
    '''

    x = int(left) + int(right) + int(top)
    x //= 3
    return x 

def calc_luminance(left,right,top):
    '''
    calculate luminace
    left = left Img (grey)
    right = right Img (grey)
    top = top Img (grey)
    '''
    lum = left.copy()
    rows,colums = left.shape

    for i in range(rows):
        for j in range(colums):
            lum[i][j] = calc_pix_luminance(left[i][j],right[i][j],top[i][j])
    return lum

def calc_bias(left,right):
    '''
    calculate bias
    left = left Img (grey)
    right = right Img (grey)
    '''
    return cv2.subtract(left,right)

def calc_bias_white(left,right):
    '''
    calculate bias and light it up
    left = left Img (grey)
    right = right Img (grey)
    '''
    return light_up_img(calc_bias(left,right),LIGHT_UP_FACTOR)
            
def clac_pix_brightness(left,right,top):
    '''
    calculate brightness for a single pixel
    left = greyValue (left Img)
    right = greyValue (right Img)
    top = greyValue (top Img)
    '''

    x = int(top - (int(left)+int(right))/2)
    if x < 0:
        x = 0
    return x

def clac_brightness(left,right,top):
    '''
    calculate brightness
    left = left Img (grey)
    right = right Img (grey)
    top = top Img (grey)
    '''

    br = left.copy()
    rows,colums = left.shape

    for i in range(rows):
        for j in range(colums):
            br[i][j] = clac_pix_brightness(left[i][j],right[i][j],top[i][j])
    return br

def clac_brightness_white(left,right,top):
    '''
    calculate brightness and light it up
    left = left Img (grey)
    right = right Img (grey)
    top = top Img (grey)
    '''

    return light_up_img(clac_brightness(left,right,top),LIGHT_UP_FACTOR)

def main():
    img_left = cv2.imread("../resource/cover_left.JPG",0)
    img_right = cv2.imread("../resource/cover_right.JPG",0)
    img_top = cv2.imread("../resource/cover_top.JPG",0)

    img_luminance = calc_luminance(img_left,img_right,img_top)
    img_bias = calc_bias(img_left,img_right)
    img_brightness = clac_brightness(img_left,img_right,img_top)
    img_bias_w = calc_bias_white(img_left,img_right)
    img_brightness_w = clac_brightness_white(img_left,img_right,img_top)

    cv2.imwrite("../out/cover_luminance.JPG",img_luminance)
    cv2.imwrite("../out/cover_bias.JPG",img_bias)
    cv2.imwrite("../out/cover_brightness.JPG",img_brightness)
    cv2.imwrite("../out/cover_bias_w.JPG",img_bias_w)
    cv2.imwrite("../out/cover_brightness_w.JPG",img_brightness_w)

    row1 = np.hstack((img_left,img_top,img_right,img_bias_w))
    row2 = np.hstack((img_luminance,img_bias,img_brightness,img_brightness_w))
        
    stacked = np.vstack((row1,row2))
    cv2.imwrite("../out/stacked.jpg",stacked)

    cv2.namedWindow('images',cv2.WINDOW_NORMAL)
    cv2.imshow("images",stacked)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
