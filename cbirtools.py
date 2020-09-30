
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
from scipy.spatial import distance as dist
from skimage.filters import gabor_kernel
from skimage import color
from scipy import ndimage as ndi
import scipy.misc, os 

class ColorDescriptor():
    """
    Defines methodes for extracting color descriptors (indexes)
    METHODS:
        getAvgs(imgPix)
        getSTDs(self, imgPix)
        getMoments(self, imgPix)
        def getHist(self, imgPix, bins=[17, 17, 17])
        def getHistHSV(self, image, bins)
        def histogram(self, image, mask, bins)
    """
    def getAvgs(self, imgPix):
        """
        Computes the mean of RGB components
        ARGUMENTS:
            imgPix: list of image pixels [[r,g,b], ...]
        :return: (avgR, avgG, avgB)
        """
        imgPix = np.array(imgPix)
        return list(imgPix.mean(axis=1).mean(axis=0))

    def getSTDs(self, imgPix):
        """
        Computes the standard deviation of RGB components
        ARGUMENTS:
            imgPix: list of image pixels [[r,g,b], ...]
        :return: (stdR, stdG, stdB)
        """
        imgPix = np.array(imgPix)
        return list(imgPix.std(axis=1).std(axis=0))
    
    def getMoments(self, imgPix):
        """
        Computes the first 3 color moments
        ARGUMENTS:
            imgPix: list of image pixels [[r,g,b], ...]
        :return: (avgR, avgG, avgB, stdR, stdG, stdB, m3R, m3G, m3B)
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
        Computes the RGB(BGR in cv2) of the image
        ARGUMENTS:
            imgPix: list of image pixels [[b,g,r], ...]
            bins: number of bins
        :return: RGB histogram
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
    
    def getHistHSV(self, image, bins=(8, 12, 3)):
        """
        Computes the HSV of the image
        ARGUMENTS:
            imgPix: list of image pixels [[h,s,v], ...]
            bins: number of bins
        :return: HSV histogram
        
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
        """
        Helper method of getHistHSV()
        computes the histograme of every region
        ARGUMENTS:
            image: the pixels of the image
            mask: the mask for regions
            bins: number of bins
        """
        # extract a 3D color histogram from the masked region of the
        # image, using the supplied number of bins per channel
        hist = cv2.calcHist([image], [0, 1, 2], mask, bins,
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




theta, frequency, sigma, bandwidth, n_slice= 8, [0.06, 0.09, 0.13, 0.18, 0.25], (1, 3, 5), (0.3, 0.7, 1), 2

def make_gabor_kernel(theta, frequency, sigma, bandwidth):
  kernels = []
  for t in range(theta):
    t = t / float(theta) * np.pi
    for f in frequency:
      if sigma:
        for s in sigma:
          kernel = gabor_kernel(f, theta=t, sigma_x=s, sigma_y=s)
          kernels.append(kernel)
      if bandwidth:
        for b in bandwidth:
          kernel = gabor_kernel(f, theta=t, bandwidth=b)
          kernels.append(kernel)
  return kernels

gabor_kernels = make_gabor_kernel(theta, frequency, sigma, None)

class Gabor(object):  

  def gabor_histogram(self, input, normalize=True):
    ''' count img histogram
  
      arguments
        input    : a path to a image or a numpy.ndarray
        normalize: normalize output histogram
  
      return
          a numpy array with size len(gabor_kernels)
    '''
    if isinstance(input, np.ndarray):  # examinate input type
      img = input.copy()
    else:
      img = cv2.imread(input) #scipy.misc
    height, width, channel = img.shape
  
    hist = self._gabor(img, kernels=gabor_kernels)
  
    if normalize:
      hist /= np.sum(hist)
  
    return hist.flatten()
  
  
  def _feats(self, image, kernel):
    '''
      arguments
        image : ndarray of the image
        kernel: a gabor kernel
      return
        a ndarray whose shape is (2, )
    '''
    feats = np.zeros(2, dtype=np.double)
    filtered = ndi.convolve(image, np.real(kernel), mode='wrap')
    feats[0] = filtered.mean()
    feats[1] = filtered.var()
    return feats
  
  
  def _power(self, image, kernel):
    '''
      arguments
        image : ndarray of the image
        kernel: a gabor kernel
      return
        a ndarray whose shape is (2, )
    '''
    image = (image - image.mean()) / image.std()  # Normalize images for better comparison.
    f_img = np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)
    feats = np.zeros(2, dtype=np.double)
    feats[0] = f_img.mean()
    feats[1] = f_img.var()
    return feats
  
  
  def _gabor(self, image, kernels=gabor_kernels, normalize=True):
  
    img = color.rgb2gray(image)
  
    results = []
    feat_fn = self._feats #self._power
    for kernel in kernels:
      results.append(self._worker(img, kernel, feat_fn))
    
    hist = np.array([res for res in results])
  
    if normalize:
      hist = hist / np.sum(hist, axis=0)
  
    return hist.T.flatten()
  
  
  def _worker(self, img, kernel, feat_fn):
    try:
      ret = feat_fn(img, kernel)
    except:
      print("return zero")
      ret = np.zeros(2)
    return ret

class Distance():
    """
    Defines different distances to use!
    """
    def manhatan(self, obj, query):
        l1, l2 = np.array(obj[1]), np.array(query[1])
        if len(l1.shape) >= 2:
            l1, l2 = l1.flatten(), l2.flatten()
        return dist.cityblock(l1,l2)

    def euclid(self, obj, query):
        l1, l2 = np.array(obj[1]), np.array(query[1])
        if len(l1.shape) >= 2:
            l1, l2 = l1.flatten(), l2.flatten()
        return dist.euclidean(l1, l2)

    def euclid_moments(self, obj, query):
        avgs1 = obj[1]
        avgs2 = query[1]
        d = 0.
        for i in range(3):
            d += ((avgs1[i]-avgs2[i])+(avgs1[i+3]-avgs2[i+3])+(avgs1[i+6]-avgs2[i+6]))**2
        return sqrt(d)
    
    def chi2_distance(self, histA, histB, eps = 1e-10):
        # compute the chi-squared distance
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
            for (a, b) in zip(histA, histB)])
        # return the chi-squared distance
        return d

    def chi(self, obj, query):
        h1 = obj[1]
        h2 = query[1]
        return cv2.compareHist(h1, h2, cv2.HISTCMP_CHISQR)

    def Bhattachatyya(self, obj, query):
        h1 = obj[1]
        h2 = query[1]
        return cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)



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