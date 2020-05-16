import cv2
import numpy as np

LVL = 4


def create_pyramid(img,lvl):
    lst =  [img]
    last_img = img

    for i in range (lvl):
        last_img = cv2.pyrDown(last_img)
        lst.append(last_img)

        cv2.imshow("hobra",last_img)
        cv2.waitKey(0)
        
    return lst

def pyramid_up(img_lst):
    img_lst = list(reversed(img_lst))
    lst = [img_lst[0]]

    for i in range(len(img_lst)-1):
        n = cv2.pyrUp(img_lst[i])
        n -= img_lst[i+1]
        lst.append(n)

        cv2.imshow("hobra",n)
        cv2.waitKey(0)
        
    return lst

def reconstruct_pyramid(img_lst):
    n = img_lst[0]
    for img in img_lst[1:]:
        n = cv2.pyrUp(n)
        k = n+img
        
        cv2.imshow("hobra",k)
        cv2.waitKey(0)
    return k


def combine_img(img_1,img_2):
    img_p1 = img_1[:,img_1.shape[1]//2:]
    img_p2 = img_2[:,:img_2.shape[1]//2]

    return np.hstack((img_p2,img_p1))

def main():
    img_1 = cv2.imread("resource/horse.png") / 255
    img_2 = cv2.imread("resource/zebra.png") / 255

    img = combine_img(img_1,img_2)

    cv2.imshow("hobra",img)
    cv2.imwrite("out/hobra.png",img*255)
    cv2.waitKey(0)

    pry_1 = create_pyramid(img_1,LVL)
    pry_2 = create_pyramid(img_2,LVL)

    pry_up_1 = pyramid_up(pry_1)
    pry_up_2 = pyramid_up(pry_2)

    c_img_lst = []
    for i1, i2 in zip(pry_up_1,pry_up_2):
        c_img = combine_img(i1,i2)
        c_img_lst.append(c_img)

        cv2.imshow("hobra",c_img)
        cv2.waitKey(0)
    
    d = reconstruct_pyramid(c_img_lst)

    cv2.imshow("hobra",d)
    cv2.imwrite("out/hobra2.png",d * 255)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()