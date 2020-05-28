import numpy as np
import cv2

neighborhood_size = 4
threshold = 0.01
dectector_free = 0.04
sobel = 5 # ungrade !
temp_size = 4

def match(out,minMaxL,w,h,img_shape,orgin_corner_point):
    point = (minMaxL[0] + img_shape[1],minMaxL[1])
    point2 = (minMaxL[0] + w + img_shape[1],minMaxL[1]  + h)
    cv2.rectangle(out,point,point2,255,1)

    pt = (minMaxL[0] + img_shape[1] + w//2 ,minMaxL[1]+ h//2)
    cv2.line(out,orgin_corner_point,pt,(0,255,0),1)

def match_template(img,out,template,orgin_corner_point,meth=cv2.TM_CCORR_NORMED):

    w, h = template.shape[::-1]

    result = cv2.matchTemplate(img,template,method=meth)
    min_a, max_a, maxL, minL = cv2.minMaxLoc(result)

    if cv2.TM_CCORR_NORMED == meth:
        match(out,minL,w,h,img.shape,orgin_corner_point)
    elif cv2.TM_SQDIFF_NORMED == meth:
        match(out,maxL,w,h,img.shape,orgin_corner_point)

def match_with_temp(point,img_in,ncc_out,sdd_out):
    temp = img_in[int(point[1] - temp_size):int(point[1] + temp_size),int(point[0] - temp_size):int(point[0] +temp_size)]
    if temp.size > 0 :
        rpt1 = (int(point[0] - temp_size),int(point[1] + temp_size))
        rpt2 = (int(point[0] + temp_size),int(point[1] - temp_size))

        cv2.rectangle(sdd_out,rpt1,rpt2,255,1)
        cv2.rectangle(ncc_out,rpt1,rpt2,255,1)

        match_template(img_right,ncc_out,temp,point,cv2.TM_CCORR_NORMED)
        match_template(img_right,sdd_out,temp,point,cv2.TM_SQDIFF_NORMED)

def show_result(img_left,img_right,coners,is_good=False,path="out/a2.png"):
    img_left_out = np.copy(img_left)
    img_left_out = cv2.cvtColor(img_left_out,cv2.COLOR_GRAY2BGR)

    img_right_out_ncc = np.copy(img_right)

    img_right_out = cv2.cvtColor(img_right_out_ncc,cv2.COLOR_GRAY2BGR)

    img_out_sdd = np.hstack((img_left_out,img_right_out))
    img_out_ncc = np.copy(img_out_sdd)
    
    if is_good:
        intrestPoints = np.int0(coners)
        for i in intrestPoints:
            x,y = i.ravel()
            pt = (x,y)
            match_with_temp(pt,img_left,img_out_ncc,img_out_sdd)
    else:
        intrestPoints = np.where(coners  >= threshold)
        for point in zip(*intrestPoints[::-1]):
            match_with_temp(point,img_left,img_out_ncc,img_out_sdd)

    img = np.vstack((img_out_ncc,img_out_sdd))

    cv2.namedWindow("img",cv2.WINDOW_NORMAL)
    cv2.imwrite(path,img)
    cv2.imshow("img",img)
    cv2.waitKey(0)

if __name__ == "__main__":

    img_left = cv2.imread("resource/church_left.png",0)
    img_right = cv2.imread("resource/church_right.png",0)

    blur_img = cv2.GaussianBlur(img_left,(3,3),0)
    #1
    img_harr_left = cv2.cornerHarris(blur_img,neighborhood_size,sobel,dectector_free)
    show_result(img_left,img_right,img_harr_left,path="out/cornerHarris.png")
    
    #gftt match nur pro Ecke (also in unmittelbarer Umgebung) nur einmal während Harris das nicht tut.

    #2 
    gftt_img = cv2.goodFeaturesToTrack(blur_img,20,0.1,1)
    show_result(img_left,img_right,gftt_img,True,path="out/goodFeaturesToTrack.png")

    #3

    orb = cv2.ORB_create(3000)
    print(orb)
    kp1, des1 = orb.detectAndCompute(img_left,None)
    kp2, des2 = orb.detectAndCompute(img_right,None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(img_left,img_right)

    matches = sorted(matches, key = lambda x:x.distance)

    orb_img = np.hstack((img_left,img_right))
    kp3, des3 = orb.detectAndCompute(orb_img,None)

    orb_img = cv2.drawMatches(img_left,kp1,img_right,kp2,matches,orb_img, flags=2)

    orb_img = cv2.drawKeypoints(orb_img,kp3,orb_img,color=(0,0,255))
    
    cv2.namedWindow("img",cv2.WINDOW_NORMAL)
    cv2.imwrite("out/orb.png",orb_img)
    cv2.imshow("img",orb_img)
    cv2.waitKey(0)
