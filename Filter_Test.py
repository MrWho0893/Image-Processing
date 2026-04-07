import cv2
import numpy as np

def Gaussfilter3 (img,sig):
    length = img.shape[0]
    width = img.shape[1]
    GaussFig = np.empty((length,width),dtype=np.uint8)
    GaussKern = np.array([[np.exp(-2/(2*sig**2)), np.exp(-1/(2*sig**2)), np.exp(-2/(2*sig**2))],
                         [np.exp(-1/(2*sig**2)), 1, np.exp(-1/(2*sig**2))],
                         [np.exp(-2/(2*sig**2)), np.exp(-1/(2*sig**2)), np.exp(2/(-2*sig**2))]])
    Sum = np.sum(GaussKern)
    GaussKern = np.divide(GaussKern,Sum)## define Gaussian Kern Array

    ##extend original figure
    GaussCal = np.vstack((np.zeros(width),img))
    GaussCal = np.vstack((GaussCal,np.zeros(width)))
    GaussCal = np.hstack((np.zeros((length+2,1)),GaussCal))
    GaussCal = np.hstack((GaussCal,np.zeros((length+2,1))))

    ## calculate Gaussian filter
    for i in range(length):
        for j in range(width):
            GaussFig[i,j] = int(np.sum(GaussCal[i:i+3,j:j+3]*GaussKern))
    return GaussFig



img = cv2.imread('Test1.png')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("orin",img)
cv2.waitKey(0)
GaussT = Gaussfilter3 (img,1.4)
Gauss = cv2.GaussianBlur(img,(3,3),0)
print(GaussT)
cv2.imshow('Gauss Test',GaussT)
cv2.waitKey(0)
cv2.imshow('Gauss',Gauss)
cv2.waitKey(0)


