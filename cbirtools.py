"""
    Contains our CBIR tools
    DESCRIPTOR EXTRACTORS
    DISTANCE FUNCTIONS
"""
from math import sqrt
import math
import numpy as np
import cv2, imutils
from mahotas import features

class ColorDescriptor():
    """
    Defines methodes for extracting descriptors (indexes)
    """
    def getAvgs(self, imgPix):
        """
        la moyenne, moment d’ordre un.
        :return: (avgR, avgG, avgB)
        """
        imgPix = np.array(imgPix)
        return list(imgPix.mean(axis=1).mean(axis=0))

    def getSTDs(self, imgPix):
        """
        L'écart type, moment d’ordre deux.
        :return:
        """
        imgPix = np.array(imgPix)
        return list(imgPix.std(axis=1).std(axis=0))
    
    def getMoments(self, imgPix):
        """
        moment d’ordre trois.
        :return:
        """
        imgPix = np.array(imgPix)
        height, width, channels = imgPix.shape
        sz = height*width
        
        mean = self.getAvgs(imgPix)
        v = [0. for _ in range(3)]
        for i in range(len(imgPix)):
            for j in range(len(imgPix[0])):
                v[0] += (imgPix[i][j][0] - mean[0])**3
                v[1] += (imgPix[i][j][1] - mean[1])**3
                v[2] += (imgPix[i][j][2] - mean[2])**3
        v = [ abs((y/sz))**(1/3) for y in v]
        features = []
        features.extend(mean)
        features.extend(self.getSTDs(imgPix))
        features.extend(v)
        return features
    
    def getHist(self, imgPix, bins=[17, 17, 17]):
        """
        Return bgr Histograms
        """
        imgPix = np.array(imgPix)
        rbgHist= cv2.calcHist(images=[imgPix], 
                            channels=[0, 1, 2], 
                            mask=None, 
                            histSize=bins, 
                            ranges=[0, 256, 0, 256, 0, 256])
        # normalize with OpenCV 2.4
        if imutils.is_cv2():
            rbgHist = cv2.normalize(rbgHist)
        # otherwise normalize with OpenCV 3+
        else:
            rbgHist = cv2.normalize(rbgHist, rbgHist)

        return rbgHist
    
    def getHistHSV(self, image, bins):
        """

        Credits to:
            https://www.pyimagesearch.com/2014/12/01/complete-guide-building-image-search-engine-python-opencv/
        """
        # convert the image to the HSV color space and initialize
        # the features used to quantify the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []
        # grab the dimensions and compute the center of the image
        (h, w) = image.shape[:2]
        (cX, cY) = (int(w * 0.5), int(h * 0.5))
        # divide the image into four rectangles/segments (top-left,
        # top-right, bottom-right, bottom-left)
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
            (0, cX, cY, h)]
        # construct an elliptical mask representing the center of the
        # image
        (axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
        ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)
        # loop over the segments
        for (startX, endX, startY, endY) in segments:
            # construct a mask for each corner of the image, subtracting
            # the elliptical center from it
            cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            cornerMask = cv2.subtract(cornerMask, ellipMask)
            # extract a color histogram from the image, then update the
            # feature vector
            hist = self.histogram(image, cornerMask, bins)
            features.extend(hist)
        # extract a color histogram from the elliptical region and
        # update the feature vector
        hist = self.histogram(image, ellipMask, bins)
        features.extend(hist)
        # return the feature vector
        return features

    def histogram(self, image, mask, bins):
        # extract a 3D color histogram from the masked region of the
        # image, using the supplied number of bins per channel
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,
            [0, 180, 0, 256, 0, 256])
        # normalize the histogram if we are using OpenCV 2.4
        if imutils.is_cv2():
            hist = cv2.normalize(hist).flatten()
        # otherwise handle for OpenCV 3+
        else:
            hist = cv2.normalize(hist, hist).flatten()
        # return the histogram
        return hist

    #_________________________________________________________________________
class TextureDescriptor:
    def getGabor(self, image):
        kernel        = cv2.getGaborKernel((21, 21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kernel       /= math.sqrt((kernel * kernel).sum())
        filtered_img  = cv2.filter2D(image,    cv2.CV_8UC3, kernel)
        heigth, width = kernel.shape 
        # convert matrix to vector 
        descriptor = cv2.resize(filtered_img, (3*width, 3*heigth), interpolation=cv2.INTER_CUBIC)
        return np.hstack(descriptor)
    
    def getGaborFilterBank(self, ksize):
        gaborFilters = []
        _lambda = [0.06, 0.09, 0.13, 0.18, 0.25]
        for lnda in _lambda:
            for tta in np.arange(0, np.pi, np.pi / 8):
                kern = cv2.getGaborKernel((ksize, ksize), 3.0, tta, lnda, 0.5, 0, ktype=cv2.CV_32F)
                gaborFilters.append(kern)
        return gaborFilters

    def getGaborFeatures(self, image):
        bank = self.getGaborFilterBank(31)
        features = []
        for kernel in bank:
            filtred = cv2.filter2D(image, cv2.CV_8UC3, kernel)
            features.append(kernel.mean())
            features.append( kernel.std())
        return features
    

    def getHaralickFeatures(self, imgPix):
        imgPix = np.array(imgPix)
        return list(features.haralick(imgPix, return_mean=True))

#________________________Shape________________________
class ShapeDescriptor:
    def getHuMoments(self, image):
        # pad the image with extra white pixels to ensure the
        # edges are not up against the borders of the image
        image[image <= 36] = 0
        thresh = cv2.copyMakeBorder(image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value = 0)
        # Perform a simple segmentation
        #(T, thresholded) = cv2.threshold(gray, 28, 255, cv2.THRESH_BINARY)
        # threshold it
        thresh[thresh > 0] = 255
        # initialize the outline image, find the outermost
        # contours (the outline) , then draw it
        outline = np.zeros(image.shape, dtype = "uint8")
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
        cv2.drawContours(outline, [cnts], -1, 255, -1)
        # return HuMoments
        return cv2.HuMoments(cv2.moments(outline)).flatten()
    

    def getZernikeMoments(self, image, radius=21):
        # pad the image with extra white pixels to ensure the
        # edges are not up against the borders of the image
        image[image <= 36] = 0
        thresh = cv2.copyMakeBorder(image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value = 0)
        # threshold it
        thresh[thresh > 0] = 255
        # initialize the outline image, find the outermost
        # contours (the outline) , then draw it
        outline = np.zeros(image.shape, dtype = "uint8")
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
        cv2.drawContours(outline, [cnts], -1, 255, -1)
        # compute Zernike moments to characterize the shape
        # of pokemon outline, then update the index
        return features.zernike_moments(outline, radius)


class Distance():
    """
    Defines different distances to use!
    """
    def manhatan(self, obj, query):
        avgs1 = obj[1]
        avgs2 = query[1]
        d = 0
        for i in range(len(avgs1)):
            d += abs(avgs1[i]-avgs2[i])
        return d

    def euclid(self, obj, query):
        avgs1 = obj[1]
        avgs2 = query[1]
        d = 0
        for i in range(len(avgs1)):
            d += (avgs1[i]-avgs2[i])**2
        return sqrt(d)

    def euclid_moments(self, obj, query):
        avgs1 = obj[1]
        avgs2 = query[1]
        d = 0.
        for i in range(3):
            d += ((avgs1[i]-avgs2[i])+(avgs1[i+3]-avgs2[i+3])+(avgs1[i+6]-avgs2[i+6]))**2
        return sqrt(d)
    
    def chi(self, obj, query):
        h1 = obj[1]
        h2 = query[1]
        return cv2.compareHist(h1, h2, cv2.HISTCMP_CHISQR)
    
    def chi2_distance(self, histA, histB, eps = 1e-10):
        # compute the chi-squared distance
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
            for (a, b) in zip(histA, histB)])
        # return the chi-squared distance
        return d

    def intersect(self, obj, query):
        h1 = obj[1]
        h2 = query[1]
        return cv2.compareHist(h1, h2, cv2.HISTCMP_INTERSECT)



class FusionDescriptors():
    
    def __init__(self, fw, sw):
        self.firstWeight = fw
        self.secondWeight = sw
        self.color = ColorDescriptor()
        self.texture = TextureDescriptor()
        self.shape = ShapeDescriptor()
    
    def getMomentsAndGabor(self, image, gray):
        combinedFeatures = []
        combinedFeatures.extend(self.color.getMoments(image)) 
        combinedFeatures.extend(self.texture.getGabor(gray))
        return combinedFeatures
    
    def getMomentsAndZernike(self, image, gray):
        combinedFeatures = []
        combinedFeatures.extend(self.color.getMoments(image)) 
        combinedFeatures.extend(self.shape.getZernikeMoments(gray))
        return combinedFeatures
        
    def manhatanDistance(self, obj, query):
        obj, query = obj[1], query[1]
        # Color feature distace
        cfd = 0
        for i in range(9):
            cfd += abs(obj[i]-query[i])
        # Texture/Shape feature distace
        tsfd = 0
        for i in range(10, len(obj)):
            tsfd += abs(obj[i]-query[i])

        return (self.firstWeight * cfd + self.secondWeight * tsfd)


#print(Descriptor.getAvgs([(2, 1, 3), (1,1,1)]))
#print(Descriptor.getSTDs([(2, 1, 3), (1,1,1)]))
#h1 = Descriptor.getHist(cv2.imread("images/1.jpg"))
#h2 = Descriptor.getHist(cv2.imread("images/52.jpg"))
"""
from PIL import ImageTk, Image
im = Image.open("images/1.jpg")
print(np.shape(list(im.getdata())))
"""