import cv2
import sys
import json
import numpy as np


def main():

    data = json.load(open("resources/camera.json"))
    img_1 = cv2.imread("resources/1.JPG")
    img_2 = cv2.imread("resources/2.JPG")
    img_3 = cv2.imread("resources/3.JPG")
    
    c_matrix = np.array(data["camera_matrix"])
    distortion = np.array(data["distortion"])
    
    img_1u = cv2.undistort(img_1,c_matrix,distortion)
    img_2u = cv2.undistort(img_2,c_matrix,distortion)
    img_3u = cv2.undistort(img_3,c_matrix,distortion)

    cv2.imwrite("out/1_undistorted.jpg",img_1u)
    cv2.imwrite("out/2_undistorted.jpg",img_2u)
    cv2.imwrite("out/3_undistorted.jpg",img_2u)

    cv2.namedWindow('images',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600,600)
    stack = np.hstack((img_1,img_1u))
    cv2.imshow('images',stack)
    cv2.waitKey(0)

    stack = np.hstack((img_2,img_2u))
    cv2.imshow('images',stack)
    cv2.waitKey(0)

    stack = np.hstack((img_3,img_3u))
    cv2.imshow('images',stack)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()