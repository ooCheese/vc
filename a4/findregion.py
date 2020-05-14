import cv2
import numpy as np

def main():
    img = cv2.imread("resource/koreanSigns.png")

    #mark region
    region = cv2.selectROI("findregion",img)

    cv2.namedWindow("findregion",cv2.WINDOW_AUTOSIZE)
    template = img[int(region[1]):int(region[1] +region[3]),int(region[0]):int(region[0] +region[2])]

    # convert to BGR to Gray
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)

    # get Template size
    w, h = template_gray.shape[::-1]

    # match with normalized cross-correlation
    result = cv2.matchTemplate(img_gray,template_gray,method=cv2.TM_CCORR_NORMED)
    
    threshold = 0.96
    loc = np.where( result >= threshold)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(img,pt,(pt[0] + w, pt[1]+ h),(0,255,0),2)

    cv2.imshow("findregion",img)
    cv2.imwrite("out/findregion.png",img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()