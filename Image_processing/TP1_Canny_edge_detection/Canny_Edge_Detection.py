import numpy as np 
import matplotlib.pyplot as plt
import scipy
import cv2
import scipy.linalg
import scipy.signal
import scipy.ndimage
import time

from Harris_Corner_Detection import partial_derivatives_xy

def img_smoothing(img,sigma) : 
    img_2 = img.copy()
    img_2 = scipy.ndimage.gaussian_filter(img_2,sigma=sigma,order=0) #phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    return(img_2)

def sobel_x(img) :
    sb_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    img_2 = scipy.signal.convolve2d(img,sb_x)
    plt.plot(img_2)
    plt.show()
    return img_2
def sobel_y(img) :
    sb_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    plt.plot(sb_y)
    img_2 = scipy.signal.convolve2d(img,sb_y)
    return img_2
def display_image(img_name,img) :
    cv2.imshow(img_name,img)

if __name__ == '__main__' :
    image = cv2.imread("CircleLineRect.png",cv2.IMREAD_GRAYSCALE)
    display_image('base_image',image)
    """step 1"""
    smooth_image = img_smoothing(img=image,sigma=1)
    display_image("smooth_image",smooth_image)
    """step 2"""
    gradient_x = sobel_x(smooth_image)
    gradient_y = sobel_y(smooth_image)
    display_image('gradient_x',gradient_x)
    display_image('gradient_y',gradient_y)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()