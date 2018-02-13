import numpy as np
import cv2
from matplotlib import pyplot as plt
import copy

img = cv2.imread('eagle_nebula.jpg',0)
hist,bins = np.histogram(img.flatten(),256,[0,256])

cdf = hist.cumsum()

cdf_normalized = cdf * hist.max()/cdf.max()

def load(imgname,color_tf):
	img = cv2.imread(imgname,color_tf)
	return img

def deep_copy(img):
        new = copy.deepcopy(img)
        return(new)

def show(img):
    cv2.imshow("image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot_cdf(img):
	plt.plot(cdf_normalized, color = "b")
	plt.hist(img.flatten(),256,[0,256], color = "r")
	plt.xlim([0,256])
	plt.legend(("cdf","histogram"),loc = "right")
	plt.show()

def eq(img):
        return(cv2.equalizeHist(img))

def fast_clahe(img):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        return(img)

#default kernel
kernel = np.ones((5,5), np.uint8)

def fast_erode(img):
        return (cv2.erode(img, kernel, iterations = 1))

def fast_dilate(img):
        return (cv2.dilate(img, kernel, iterations = 1))

def create_kernel(r,c):
        kernel = np.ones((r,c), np.uint8)
        return kernel


def fast_threshold(img,choice):
        if choice == 1:
                ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
                return th1
        elif choice == 2:
                th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
                return th2
        else:
                th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
                return th3

def find_contours(img):
        ret,thresh = cv2.threshold(img,127,255,0)
        img, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        return contours

def draw_contours(source,new,contours):
        cv2.drawContours(new, contours, -1, (0,255,0), 3)
        return new
        
        



"""
show(img)
plot_cdf(img)
"""
