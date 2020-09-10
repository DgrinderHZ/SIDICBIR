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
        #t = [0. for _ in range(3)]
        #for i in range(len(imgPix)):
        #    for j in range(len(imgPix[0])):
        #        t[0] += imgPix[i][j][0]
        #        t[1] += imgPix[i][j][1]
        #        t[2] += imgPix[i][j][2]
        #t = [y/(len(imgPix)*len(imgPix[0])) for y in t]
        #print(imgPix.shape)
        #print(imgPix.mean(axis=1).mean(axis=0), "equals", t)
        return list(imgPix.mean(axis=1).mean(axis=0))

    def getSTDs(imgPix):
        """
        L'écart type, moment d’ordre deux.
        :return:
        """
        imgPix = np.array(imgPix)
        """
        mean = Descriptor.getAvgs(imgPix)
        v = [0. for _ in range(3)]
        for i in range(len(imgPix)):
            for j in range(len(imgPix[0])):
                v[0] += (imgPix[i][j][0] - mean[0])**2
                v[1] += (imgPix[i][j][1] - mean[1])**2
                v[2] += (imgPix[i][j][2] - mean[2])**2
        v = [ sqrt(y/(len(imgPix)*len(imgPix[0])))  for y in v]
        """
        #print(v, "equals", imgPix.std(axis=1).std(axis=0))
        return list(imgPix.std(axis=1).std(axis=0))
    
    def getCMoments(imgPix):
        """
        moment d’ordre trois.
        :return:
        """
        imgPix = np.array(imgPix)
        mean = Descriptor.getAvgs(imgPix)
        v = [0. for _ in range(3)]
        for i in range(len(imgPix)):
            for j in range(len(imgPix[0])):
                v[0] += (imgPix[i][j][0] - mean[0])**3
                v[1] += (imgPix[i][j][1] - mean[1])**3
                v[2] += (imgPix[i][j][2] - mean[2])**3
        v = [ abs(y/(len(imgPix)*len(imgPix[0])))**(1/3) for y in v]
        features = []
        features.extend(mean)
        features.extend(Descriptor.getSTDs(imgPix))
        features.extend(v)
        return features
    
    def getHist(imgPix):
        """
        Return bgr Histograms
        """
        imgPix = np.array(imgPix)
        rbgHist= cv2.calcHist(images=[imgPix], 
                            channels=[0, 1, 2], 
                            mask=None, 
                            histSize=[17, 17, 17], 
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

    def euclid_moments(obj, query):
        avgs1 = obj[1]
        avgs2 = query[1]
        d = 0.
        for i in range(3):
            d += ((avgs1[i]-avgs2[i])+(avgs1[i+3]-avgs2[i+3])+(avgs1[i+6]-avgs2[i+6]))**2
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