import numpy as np
import math
import cv2

rotation_angle = 0
scale = 1
gauss_kernel_size = 1
neighborhood_size = 1
threshold = 0.97
dectector_free = 0.04
sobel = 1

resource_img = cv2.imread("resource/church_left.png")
img = resource_img

def calc_img():
    global img

    img = resource_img
    img = cv2.GaussianBlur(img,(gauss_kernel_size,gauss_kernel_size),0)

    rot_matrix = cv2.getRotationMatrix2D(tuple(np.array(img.shape[1::-1])//2),rotation_angle,scale)
    img = cv2.warpAffine(img,rot_matrix,img.shape[1::-1], flags=cv2.INTER_LINEAR)

    harr_img = calc_harr_coner(img)
    harr_img = cv2.cvtColor(harr_img,cv2.COLOR_BGR2GRAY)

    brightPoints = np.where(harr_img >= threshold)

    for point in zip(*brightPoints[::-1]):
        cv2.circle(img, point, 2, (200,0,0), 1)

    cv2.imshow("cornes",img)
    cv2.imwrite("out/test.jpg",img)

def calc_harr_coner_gray(img):
    return cv2.cornerHarris(img,neighborhood_size,sobel,dectector_free)

def calc_harr_coner(img):
    b , g, r = cv2.split(img)

    b = calc_harr_coner_gray(b)
    g = calc_harr_coner_gray(g)
    r = calc_harr_coner_gray(r)

    return cv2.merge((b,g,r))

def on_slider_gaus_kernel_size(val):
    global gauss_kernel_size
    gauss_kernel_size = val

    if val % 2 == 0:
        gauss_kernel_size += 1
    
    calc_img()

def on_slider_neighborhood_size(val):
    global neighborhood_size
    neighborhood_size = val
    calc_img()

def on_slider_threshold(val):
    global threshold
    threshold = val/100
    calc_img()


def on_slider_dectector_free(val):
    global dectector_free
    dectector_free = val/100 * 0.02 + 0.04
    calc_img()

def on_slider_sobel(val):
    global sobel
    sobel = val

    if val % 2 == 0:
        sobel += 1

    calc_img()

def on_slider_scale(val):
    global scale
    scale = val/100
    calc_img()

def on_slider_rotate(val):
    global rotation_angle
    rotation_angle = val
    calc_img()

if __name__ == "__main__":

    #load img

    cv2.namedWindow("cornes",cv2.WINDOW_GUI_NORMAL)

    cv2.createTrackbar("rotation angle","cornes",0,360,on_slider_rotate)
    cv2.createTrackbar("scale angle","cornes",100,200,on_slider_scale)
    cv2.createTrackbar("gauss kernel","cornes",1,100,on_slider_gaus_kernel_size)
    cv2.createTrackbar("neighborhood size","cornes",1,100,on_slider_neighborhood_size)
    cv2.createTrackbar("sobel","cornes",1,31,on_slider_sobel)
    cv2.createTrackbar("threshold in %","cornes",97,100,on_slider_threshold)
    cv2.createTrackbar("dectector free (0 => 0.4 , 100 => 0.6)","cornes",0,100,on_slider_dectector_free)

    img = calc_img()
    cv2.waitKey(0)

