import numpy as np 
import math
import cv2

def fftshift(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return np.log(np.abs(fshift))

def fft(img):
    f = np.fft.fft2(img)
    return np.log(np.abs(f))

def ifft(img):
    f_ishift = np.fft.ifftshift(img)
    img_back = np.fft.ifft2(f_ishift)
    return np.real(img_back)

def calcG(x,y,sigma):
    return (1/(2*math.pi*sigma**2))*math.e**(-(x**2+y**2)/(2*sigma**2))

def G(img,sigma):
    n = np.copy(img)
    for x in range(n.shape[0]):
        for y in range (n.shape[1]):
            n[x][y] = calcG(x,y,sigma)
    return n

def gen_hybrid(i_1,i_2,sigma):
    #I_1 = fftshift(i_1)
    #I_2 = fftshift(i_2)

    G_1 = cv2.GaussianBlur(i_1,(sigma,sigma),0)
    G_2 = cv2.GaussianBlur(i_2,(sigma,sigma),0)

    return i_1 * G_1 + i_2*(1-G_2)


def main():
    i_1 = cv2.imread("resource/Marylin_grey.png",0) / 255 #gray
    i_2 = cv2.imread("resource/John_grey.png",0) / 255 #gray
    sigma = 11

    H_gray = gen_hybrid(i_1,i_2,sigma)

    cv2.imshow("hybrid gray",H_gray)
    cv2.imwrite("out/hybrid_gray.png",H_gray* 255)
    cv2.waitKey(0)

    #color hybrid
    i_1 = cv2.imread("resource/pizza1.jpg") / 255 #color
    i_2 = cv2.imread("resource/pizza2.jpg") / 255 #color

    #resize
    width = int(i_1.shape[1] / 10)
    height = int(i_2.shape[0] / 10)
    dim = (width, height)
    i_1 = cv2.resize(i_1,dim)
    i_2 = cv2.resize(i_2,dim)

    H_col = gen_hybrid(i_1,i_2,sigma)

    cv2.imshow("hybrid",H_col)
    cv2.imwrite("out/hybrid_color.png",H_col* 255)
    cv2.waitKey(0)
    


if __name__ == "__main__":
    main()