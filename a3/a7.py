import cv2
import numpy as np
import sys



src_img = cv2.imread("resource/cheese.jpg")
src_img = cv2.resize(src_img,(400,400))

sigma_x = 0
ddepth = -1
ksize = 1
diameter = 1
sigma_color = 1
simga_space = 1
canny_th1 = 0
canny_th2 = 10

#2
gray_img = cv2.cvtColor(src_img,cv2.COLOR_BGR2GRAY)
sharr_x = cv2.Scharr(gray_img,ddepth,1,0)
sharr_y = cv2.Scharr(gray_img,ddepth,0,1)
len_img = np.copy(gray_img)
#geht bestimmt schoener
for x in range(len_img.shape[0]):
    for y in range(len_img.shape[1]):
        len_img[x][y] = np.linalg.norm([sharr_x[x][y],sharr_y[x][y]])

sharr_x = cv2.cvtColor(sharr_x,cv2.COLOR_GRAY2BGR)
sharr_y = cv2.cvtColor(sharr_y,cv2.COLOR_GRAY2BGR)
len_img = cv2.cvtColor(len_img,cv2.COLOR_GRAY2BGR)


def on_slide_sigma_x(val):
    global sigma_x

    sigma_x = int(val)
    calc_img()

def on_slide_cannyth1(val):
    global canny_th1

    canny_th1 = int(val)
    calc_img()

def on_slide_cannyth2(val):
    global canny_th2

    canny_th2 = int(val)
    calc_img()

def on_slide_sigma_color(val):
    global sigma_color

    sigma_color = float(val)
    calc_img()

def on_slide_sigma_space(val):
    global simga_space

    simga_space = float(val)
    calc_img()

def on_slide_diameter(val):
    global diameter

    diameter = int(val)
    calc_img()

def on_slide_ksize(val):
    global ksize

    ksize = int(val)
    calc_img()

def calc_img():

    if ksize % 2 == 0 :
        gaus_ksize = ksize+1
    else:
        gaus_ksize = ksize

    box_img = cv2.boxFilter(src_img,ddepth,(ksize,ksize))
    gaus_img =cv2.GaussianBlur(src_img,(gaus_ksize,gaus_ksize),sigma_x)
    bi_img = cv2.bilateralFilter(src_img,diameter,sigma_color,simga_space)
    med_img = cv2.medianBlur(src_img,gaus_ksize)
    #3
    canny = cv2.Canny(src_img,canny_th1,canny_th2)
    canny = cv2.merge((canny,canny,canny))

    row1 = np.hstack((box_img,gaus_img,bi_img,med_img))
    row2 = np.hstack((sharr_x,sharr_y,len_img,canny))
    stack = np.vstack((row1,row2))
    cv2.imshow('aufg 7',stack)

def main():
    global src_img

    cv2.namedWindow("aufg 7",cv2.WINDOW_NORMAL)
    cv2.createTrackbar("ksize","aufg 7",1,100,on_slide_ksize)
    cv2.createTrackbar("sigma x","aufg 7",1,100,on_slide_sigma_x)
    cv2.createTrackbar("sigma color","aufg 7",1,100,on_slide_sigma_color)
    cv2.createTrackbar("sigma space","aufg 7",1,100,on_slide_sigma_space)
    cv2.createTrackbar("sigma diameter","aufg 7",1,100,on_slide_diameter)
    cv2.createTrackbar("canny threshold 1","aufg 7",1,1000,on_slide_cannyth1)
    cv2.createTrackbar("canny threshold 2","aufg 7",1,1000,on_slide_cannyth2)
    calc_img()
    cv2.waitKey(0)

if __name__ == "__main__":
    main()