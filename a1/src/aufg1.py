import cv2
import numpy as np

def calc_pix_luminance(left,right,top):
    x = int(left) + int(right) + int(top)
    x //= 3
    return x 

def calc_luminance(left,right,top):
    lum = left.copy()
    rows,colums = left.shape

    for i in range(rows):
        for j in range(colums):
            lum[i][j] = calc_pix_luminance(left[i][j],right[i][j],top[i][j])
    return lum

def calc_bias(left,right):
    return cv2.subtract(left,right)

def calc_bias_white(left,right,top):
    bias = calc_bias(left,right)
    
    print(bias)
    return (np.full(top.shape, 0.6) - np.negative(bias)).astype(np.uint8)
            
def clac_pix_brightness(left,right,top):
    x = int(top - (int(left)+int(right))/2)
    if x < 0:
        x = 0
    return x

def clac_brightness(left,right,top):
    br = left.copy()
    rows,colums = left.shape

    for i in range(rows):
        for j in range(colums):
            br[i][j] = clac_pix_brightness(left[i][j],right[i][j],top[i][j])
    return br

def main():
    img_left = cv2.imread("a1/resource/cover_left.JPG",0)
    img_right = cv2.imread("a1/resource/cover_right.JPG",0)
    img_top = cv2.imread("a1/resource/cover_top.JPG",0)

    img_luminance = calc_luminance(img_left,img_right,img_top)
    img_bias = calc_bias(img_left,img_right)
    img_brightness = clac_brightness(img_left,img_right,img_top)
    img_bias_w = calc_bias_white(img_left,img_right,img_top)

    cv2.imwrite("a1/out/cover_luminance.JPG",img_luminance)
    cv2.imwrite("a1/out/cover_bias.JPG",img_bias)
    cv2.imwrite("a1/out/cover_brightness.JPG",img_brightness)
    cv2.imwrite("a1/out/cover_bias_w.JPG",img_bias_w)

    row1 = np.hstack((img_left,img_top,img_right,img_bias_w))
    row2 = np.hstack((img_luminance,img_bias,img_brightness,img_bias_w))
        
    stacked = np.vstack((row1,row2))

    cv2.namedWindow('images',cv2.WINDOW_NORMAL)
    cv2.imshow("images",stacked)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
