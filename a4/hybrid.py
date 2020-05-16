import numpy as np
from matplotlib import pyplot
import scipy
import math
import cv2


OUT_DIR = "out/hybrid/"

def fftshift(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return fshift

def ifft(img):
    f_ishift = np.fft.ifftshift(img)
    img_back = np.fft.ifft2(f_ishift)
    return np.real(img_back)

def filter(img,sigma):
    out = np.zeros(img.shape)
    h = img.shape[0]
    w = img.shape[1]

    H = h // 2
    W = w // 2

    A = 1/sigma

    for x in range(h):
        for y in range(w):
            #out[x][y]= math.exp(-1.0*((H-x)**2 + (W-y)**2)/(A**2))
            out[x][y] = (1/(2*math.pi*sigma**2))*math.e**(-1.0 * (((y - W)**2 + (x - H)**2)/(2 * sigma**2)))
    return out

def gen_color_hybrid(i_1,i_2,sigma):
    #split images in  b g r channels 
    b1 , g1, r1 = cv2.split(i_1)
    b2 , g2, r2 = cv2.split(i_2)

    B = gen_hybrid(b1,b2,sigma,"colorFilter/hybrid_color_blue_filter")
    G = gen_hybrid(g1,g2,sigma,"colorFilter/hybrid_color_green_filter")
    R = gen_hybrid(r1,r2,sigma,"colorFilter/hybrid_color_red_filter")

    return cv2.merge((B,G,R))


def gen_hybrid(i_1,i_2,sigma,name="hybrid"):
    I_1 = fftshift(i_1)
    I_2 = fftshift(i_2)

    #G_1 = cv2.GaussianBlur(I_1,(sigma,sigma),0)
    G_1 = filter(I_1,sigma) # low pass
    G_1 = np.interp(G_1,(np.min(G_1),np.max(G_1)),(0,1))
    G_2 = 1 - G_1 # high pass

    H1 = ifft(I_1 * G_1)
    H2 = ifft(I_2 * G_2)

    cv2.imshow("hybrid",H1)
    cv2.imwrite(OUT_DIR+name+"_1.png",H1 * 255)
    cv2.waitKey(0)

    cv2.imshow("hybrid",H2)
    cv2.imwrite(OUT_DIR+name+"_2.png",H2 * 255)
    cv2.waitKey(0)

    H = I_1 * G_1 + I_2*G_2
    return ifft(H)

def main():
    np.set_printoptions(suppress=True)

    i_1 = cv2.imread("resource/Marylin_grey.png",0) / 255 #gray
    i_2 = cv2.imread("resource/John_grey.png",0) / 255 #gray
    sigma = 21

    H_gray = gen_hybrid(i_1,i_2,sigma,"GrayFilter/hybrid_gray_filter")

    cv2.imshow("hybrid",H_gray)
    cv2.imwrite(OUT_DIR+"hybrid_gray.png",H_gray * 255)
    cv2.waitKey(0)

    #color hybrid 
    i_1 = cv2.imread("resource/horse.png") / 255 #color
    i_2 = cv2.imread("resource/zebra.png") / 255 #color

    H_col = gen_color_hybrid(i_1,i_2,sigma)

    cv2.imshow("hybrid",H_col)
    cv2.imwrite(OUT_DIR+"hybrid_color.png",H_col * 255)
    cv2.waitKey(0)
    


if __name__ == "__main__":
    main()