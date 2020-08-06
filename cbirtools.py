"""
    Contains our CBIR tools
    DESCRIPTOR EXTRACTORS
    DISTANCE FUNCTIONS
"""
from math import sqrt
import numpy as np
import cv2

class Descriptor():
    """
    Defines methodes for extracting descriptors (indexes)
    """
    def getAvgs(imgPix):
        """
        la moyenne, moment d’ordre un.
        :return: (avgR, avgG, avgB)
        """
        imgPix = np.array(imgPix)
        #print(imgPix.shape)
        #print(imgPix.mean(axis=1).mean(axis=0), "hh")
        return list(imgPix.mean(axis=1).mean(axis=0))

    def getSTDs(imgPix):
        """
        la moyenne, moment d’ordre deux.
        :return:
        """
        imgPix = np.array(imgPix)
        return list(imgPix.std(axis=0))

    def getHist(imgPix):
        """
        Return bgr Histograms
        """
        imgPix = np.array(imgPix)
        rbgHist= cv2.calcHist(images=[imgPix], 
                            channels=[0, 1, 2], 
                            mask=None, 
                            histSize=[8, 8, 8], 
                            ranges=[0, 256, 0, 256, 0, 256])
		
        return rbgHist









class Distance():
    """
    Defines different distances to use!
    """
    def euclid(obj, query):
        avgs1 = obj[1]
        avgs2 = query[1]
        d = 0
        for i in range(len(avgs1)):
            d += (avgs1[i]-avgs2[i])**2
        return sqrt(d)
    
    def chi(obj, query):
        h1 = obj[1]
        h2 = query[1]
        return cv2.compareHist(h1, h2, cv2.HISTCMP_CHISQR)

    def intersect(obj, query):
        h1 = obj[1]
        h2 = query[1]
        return cv2.compareHist(h1, h2, cv2.HISTCMP_INTERSECT)
#print(Descriptor.getAvgs([(2, 1, 3), (1,1,1)]))
#print(Descriptor.getSTDs([(2, 1, 3), (1,1,1)]))
#h1 = Descriptor.getHist(cv2.imread("images/1.jpg"))
#h2 = Descriptor.getHist(cv2.imread("images/52.jpg"))
"""
from PIL import ImageTk, Image
im = Image.open("images/1.jpg")
print(np.shape(list(im.getdata())))
"""