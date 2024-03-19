from scipy.ndimage import convolve
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np

def gaussian_kernel(size, sigma):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def sobel_filters(img):
    gx = np.zeros((len(img),len(img[0])))
    gy = np.zeros((len(img),len(img[0])))
    g = np.zeros((len(img),len(img[0])))
    theta = np.zeros((len(img),len(img[0])))

    for i in range(1, len(img)-1): # Height
        for j in range(1, len(img[0]) - 1): # Width
            gx[i][j] = (img[i - 1][j - 1] + 2*img[i][j - 1] + img[i + 1][j - 1]) - (img[i - 1][j + 1] + 2*img[i][j + 1] + img[i +
                        1][j + 1])
            gy[i][j] = (img[i - 1][j - 1] + 2*img[i - 1][j] + img[i - 1][j + 1]) - (img[i + 1][j - 1] + 2*img[i + 1][j] + img[i +
                        1][j + 1])
            g[i][j] = min(255, np.sqrt(gx[i][j]**2 + gy[i][j]**2))
            theta[i][j] = np.arctan2(gy[i][j], gx[i][j])

    return (g, theta)


def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180
    q = 255
    r = 255
    for i in range(1,M-1):
        for j in range(1,N-1):
                
                
               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0
    
    return Z

def threshold(img):
    res = np.zeros_like(img, dtype=np.int32)
    
    highThreshold = 80
    lowThreshold = 20

    weak = np.int32(25)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(img > highThreshold)
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return (res, weak, strong)

def hysteresis(img, weak, strong):
     
    for i in range(1, len(img) - 1): 
        for j in range(1, len(img[0]) - 1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

img = Image.open("data/lena.jpg")
img = img.convert("L")
img = np.array(img)

img_filtered = convolve(img, gaussian_kernel(5, sigma=1.4))
grad, theta = sobel_filters(img_filtered)
img_nms = non_max_suppression(grad, theta)
img_thresh, weak, strong = threshold(img_nms)
img_final = hysteresis(img_thresh, weak, strong)  
plt.subplot(121)
plt.imshow(img_thresh, cmap='gray')
plt.title("Double thresholding")
plt.axis('off')

plt.subplot(122)
plt.imshow(img_final, cmap='gray')
plt.title("Edge Tracking by Hysteresis")
plt.axis('off')
plt.show()