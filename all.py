

"""
    Contains our CBIR tools
    DESCRIPTOR EXTRACTORS
    DISTANCE FUNCTIONS
"""


from PIL import ImageTk
import PIL.Image
import cv2
from sklearn.metrics import confusion_matrix
from glob import glob
import numpy as np
from os.path import exists
from os import mkdir
from math import sqrt
import  csv
import  imutils
from mahotas import features
from scipy.spatial import distance as dist
from skimage.filters import gabor_kernel
from skimage import color
from scipy import ndimage as ndi
import abc
from heapq import heappush, heappop
import collections
from itertools import combinations, islice
from tkinter import *
from tkinter import filedialog, messagebox





def writeCSV_AVG(csvFile, data):
    """
    Writes indexes to index database.
    ARGUMENTS:
        csvFile: filename
        data = [Filename, avg]
    """
    with open(csvFile, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(data)

def readCSV_AVG(csvFile):
    """
    Reads indexes from index database.
    RETURN:
    [Filename, avg]
    """
    result = []
    with open(csvFile, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in spamreader:
            result.append(row)
    for i in range(1, len(result[0])):
        result[0][i] = float(result[0][i])
    return result[0]

# ********************** TESTING *********************** #
'''
data = ['default/2.jpg', 73.67599487304685, 108.76373291015615, 72.97486368815103]
writeCSV_AVG('eggs.csv', data)
print(readCSV_AVG('eggs.csv'))
'''




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
      #print("return zero")
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
        self.texture = Gabor() 
        self.shape = ShapeDescriptor()
    
    def getMomentsAndGabor(self, image, imgFile):
        combinedFeatures = []
        c = self.color.getMoments(image)
        g = self.texture.gabor_histogram(imgFile)
        combinedFeatures.extend(self.normalize(c))
        combinedFeatures.extend(self.normalize(g))
        return combinedFeatures
    
    def getMomentsAndZernike(self, image, gray):
        combinedFeatures = []
        c = self.color.getMoments(image)
        z = self.shape.getZernikeMoments(gray)
        combinedFeatures.extend(self.normalize(c))
        combinedFeatures.extend(self.normalize(z))
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

    def normalize(self, vect):
        """
        Normalise
        :param vect:
        :return:
        """
        m, s = np.mean(vect), np.std(vect)
        return [(x-m)/s for x in vect]



#print(Descriptor.getAvgs([(2, 1, 3), (1,1,1)]))
#print(Descriptor.getSTDs([(2, 1, 3), (1,1,1)]))
#h1 = Descriptor.getHist(cv2.imread("images/1.jpg"))
#h2 = Descriptor.getHist(cv2.imread("images/52.jpg"))
"""
from PIL import ImageTk, Image
im = Image.open("images/1.jpg")
print(np.shape(list(im.getdata())))
"""



#_____________________________________ Algorithmes de promotion et de partition ___________________________________#
def M_LB_DIST_confirmed(entries, current_routing_entry, d):
    """Promotion algorithm. Maximum Lower Bound on DISTance. Confirmed.
    Return the object that is the furthest apart from current_routing_entry
    using only precomputed distances stored in the entries.

    This algorithm does not work if current_routing_entry is None and the
    distance_to_parent in entries are None. This will happen when handling
    the root. In this case the work is delegated to M_LB_DIST_non_confirmed.

    arguments:
        entries N: set of entries from which two routing objects must be promoted.

        current_routing_entry: the routing_entry that was used
        for the node containing the entries previously.
        None if the node from which the entries come from is the root.

        d: distance function.
    """
    if current_routing_entry is None or any(e.distance_to_parent is None for e in entries):
        return M_LB_DIST_non_confirmed(entries,current_routing_entry,d)

    # if entries contain only one element or two elements twice the same, then
    # the two routing elements returned could be the same. (could that happen?)
    new_entry = max(entries, key=lambda e: e.distance_to_parent)
    return (current_routing_entry.obj, new_entry.obj)


def M_LB_DIST_non_confirmed(entries, unused_current_routing_entry, d):
    """Promotion algorithm. Maximum Lower Bound on DISTance. Non confirmed.
    Compares all pair of objects (in entries) and select the two who are
    the furthest apart.
    """
    objs = map(lambda e: e.obj, entries)
    return max(combinations(objs, 2), key=lambda two_objs: d(*two_objs))


#If the routing objects are not in entries it is possible that
#all the elements are in one set and the other set is empty.
def generalized_hyperplane(entries, routing_object1, routing_object2, d):
    """Partition algorithm.
    Each entry is assigned to the routing_object to which it is the closest.
    This is an unbalanced partition strategy.
    Return a tuple of two elements. The first one is the set of entries
    assigned to the routing_object1 while the second is the set of entries
    assigned to the routing_object2"""
    partition = (set(), set())
    for entry in entries:
        partition[d(entry.obj, routing_object1) > d(entry.obj, routing_object2)].add(entry)

    if not partition[0] or not partition[1]:
        # TODO
        partition = (set(islice(entries, len(entries)//2)),set(islice(entries, len(entries)//2, len(entries))))

    return partition


#_____________________________________ Classe M-Tree ___________________________________#
class MTree(object):
    """
    MTree(d, max_nodes): Class principal pour instancier une structure de donnees M-Tree

    d: distance utilisee
    max_nodes: M
    ____________________________________
    ____________________________________
    """
    def __init__(self, d, max_nodes=4, promote=M_LB_DIST_confirmed, partition=generalized_hyperplane):
        """
        Create a new MTree.
        Arguments:
        d: distance function.
        max_node_size: M optional. Maximum number of entries in a node of
            the M-tree
        promote: optional. Used during insertion of a new value when
            a node of the tree is split in two.
            Determines given the set of entries which two entries should be
            used as routing object to represent the two nodes in the
            parent node.
            This is delving pretty far into implementation choices of
            the Mtree. If you don't understand all this just swallow
            the blue pill, use the default value and you'll be fine.
        partition: optional. Used during insertion of a new value when
            a node of the tree is split in two.
            Determines to which of the two routing object each entry of the
            split node should go.
            This is delving pretty far into implementation choices of
            the Mtree. If you don't understand all this just swallow
            the blue pill, use the default value and you'll be fine.
        """
        if not callable(d):
            #Why the hell did I put this?
            #This is python, we use dynamic typing and assumes the user
            #of the API is smart enough to pass the right parameters.
            raise TypeError('d is not a function')
        if max_nodes < 2:
            raise ValueError('max_node_size must be >= 2 but is %d' % max_nodes)
        self.d = d
        self.max_nodes = max_nodes
        self.promote = promote
        self.partition = partition
        self.size = 0
        self.root = LeafNode(self)


    def __len__(self):
        """
            Return taille de l'arbre
        """
        return self.size

    def add(self, obj):
        self.root.add(obj)
        self.size += 1

    def add_all(self, iterable):
        for obj in iterable:
            self.add(obj)

    def k_NN_search(self, query_obj, k=1):
        """
        Methode k_NN_search:
            Algorithme de recherche K-plus proches voisins
        Args:
            query_obj: Q
            k: nombre de voisins

        Return the k objects the most similar to query_obj.
        Implementation of the k-Nearest Neighbor algorithm.
        Returns a list of the k closest elements to query_obj, ordered by
        distance to query_obj (from closest to furthest).
        If the tree has less objects than k, it will return all the
        elements of the tree.
        """
        k = min(k, len(self)) # on prend k < la taille de M-tree
        if k == 0: return []
        pr = []
        # Initialiser pr avec le Noeud root et dmin et dquery a 0
        heappush(pr, PrEntry(self.root, 0, 0))
        # Un objet qui va contenir les k plus proches voisins
        nn = NN(k)
        while pr:
            # NextNode = ChooseNode(PR);
            prEntry = heappop(pr)
            if(prEntry.dmin > nn.search_radius()):
                # best candidate is too far, we won't have better a answer
                # we can stop
                break
            # k_NN_NodeSearch(NextNode,Q,k);
            prEntry.tree.search(query_obj, pr, nn, prEntry.d_query)

        # Return the final result
        return nn.result_list()

    def range_search(self, query_obj, r=5):
        """
        Methode range_search:
            Algorithme de recherche K-plus proches voisins
        Args:
            query_obj: Q
            k: nombre de voisins

        Return the k objects the most similar to query_obj.
        Implementation of the k-Nearest Neighbor algorithm.
        Returns a list of the k closest elements to query_obj, ordered by
        distance to query_obj (from closest to furthest).
        If the tree has less objects than k, it will return all the
        elements of the tree.
        """
        if r == 0: return []
        pr = []
        # Initialiser pr avec le Noeud root et dmin et dquery a 0
        heappush(pr, PrEntry(self.root, 0, 0))
        # Un objet qui va contenir les k plus proches voisins
        # Un objet qui va contenir les k plus proches voisins
        nn = NN(100)
        while pr:
            # NextNode = ChooseNode(PR);
            prEntry = heappop(pr)
            if(prEntry.dmin > r):
                # best candidate is too far, we won't have better a answer
                # we can stop
                break
            # k_NN_NodeSearch(NextNode,Q,k);
            prEntry.tree.rangeSearch(query_obj, pr, nn, prEntry.d_query, r)
        # Return the final result
        return nn.result_list()






#_____________________________________ k NN array class __________________________________#

NNEntry = collections.namedtuple('NNEntry', 'obj dmax')

class NN(object):
    """
    a k-elements array, NN, which, at the end of execution,
    will contain the result.
    NN[i] = [oid(Oj),d(Oj, Q)], i = 1,...,k
    """
    def __init__(self, size):

        self.elems = [NNEntry(None, float("inf"))] * size
        #store dmax in NN as described by the paper
        #but it would be more logical to store it separately
        self.dmax = float("inf")

    def __len__(self):
        return len(self.elems)

    def search_radius(self):
        """The search radius of the knn search algorithm.
             aka dmax
             The search radius is dynamic.
        """
        return self.dmax

    def update(self, obj, dmax):
        """
        performs an ordered insertion in the NN array

        can change the search radius dmax: dk = NN_Update([_, dmax(T (Or))]);
        """
        # Si objet et null, update dmax
        if obj == None:
            # Internal node case
            self.dmax = min(self.dmax, dmax)
            return
        # Sinon, ajouter obj et ordonner
        # Leaf node case: We perform the insertion
        self.elems.append(NNEntry(obj, dmax)) # we have now k+1 elements
        # Sort
        for i in range(len(self)-1, 0, -1):
            if self.elems[i].dmax < self.elems[i-1].dmax:
                self.elems[i-1], self.elems[i] = self.elems[i], self.elems[i-1]
            else:
                break
        # we pop the last element so that we return to k
        # elements state
        self.elems.pop()

    # TODO: Update this to return oid alone!
    def result_list(self):
        """
        Retourne les objets de NN, les oid
        """
        result = list(map(lambda entry: entry.obj[0] if entry.obj else "None", self.elems))
        return result

#_____________________________________ PR class __________________________________
# PR is a queue of pointers to active sub-trees, i.e. sub-trees where
# qualifying objects can be found. With the pointer to (the root of) sub-tree T (Or),
# a lower bound, dmin(T(Or)), on the distance of any object in T(Or) from Q is also kept.

class PrEntry(object):
    def __init__(self, tree, dmin, d_query):
        """
        Constructor.
        arguments:
            tree: ptr(T(Or))
            dmin: dmin(T(Or))
            d_query: distance d to searched query object
        """
        self.tree = tree
        self.dmin = dmin
        self.d_query = d_query

    def __lt__(self, other):
        return self.dmin < other.dmin

    def __repr__(self):
        return "PrEntry(tree:%r, dmin:%r)" % (self.tree, self.dmin)




# <<<<<<<<<<<<<<<<< Done!
#_____________________________________ Classes Entry: Data (keys: Btree) ____________________________________#
class MTEntry(object): # Entry == Object
    """

    The leafs and internal nodes of the M-tree contain a list of instances of
    this class.

    The distance to the parent is None if the node in which this entry is
    stored has no parent (root).

    radius and subtree are None if the entry is contained in a leaf.
    Used in set and dict even tough eq and hash haven't been redefined

    Common attributs:
        obj: Oj or Or
        distance_to_parent: d(O,P(O))
    Leaf node attributs:
        not defined here : oid (Will be defined later for images)
    Internal node (non-leaf) attributs:
        radius: Covering radius r(Or).
        subtree: Pointer to covering tree T(Or).
    """
    def __init__(self, obj, distance_to_parent=None, radius=None, subtree=None):
        self.obj = obj
        self.distance_to_parent = distance_to_parent
        self.radius = radius
        self.subtree = subtree


#_____________________________________ Classes Nodes ___________________________________________#


class AbstractNode(object):
    """An abstract leaf of the M-tree.
    Concrete class are LeafNode and InternalNode

    We need to keep a reference to mtree so that we can know if a given node
    is root as well as update the root.

    We need to keep both the parent entry and the parent node (i.e. the node
    in which the parent entry is) for the split operation. During a split
    we may need to remove the parent entry from the node as well as adding
    a new entry to the node."""
    __metaclass__ = abc.ABCMeta

    def __init__(self, mtree, parent_node=None, parent_entry=None, entries=None):
        #There will be an empty node (entries set is empty) when the tree
        #is empty and there only is an empty root.
        #May also be empty during construction (create empty node then add
        #the entries later).
        self.mtree = mtree
        self.parent_node = parent_node # Np
        self.parent_entry = parent_entry
        self.entries = set(entries) if entries else set() # NRO/NO

    def __repr__(self): # pragma: no cover
        #entries might be big. Only prints the first few elements
        entries_str = '%s' % list(islice(self.entries, 2))
        if len(self.entries) > 2:
            entries_str = entries_str[:-1] + ', ...]'

        return "%s(parent_node: %s, parent_entry: %s, entries:%s)" % (
            self.__class__.__name__,
            self.parent_node.repr_class() \
                if self.parent_node else self.parent_node,
            self.parent_entry,
            entries_str)

    def repr_class(self): # pragma: no cover
        return self.__class__.__name__ + "()"

    def __len__(self):
        return len(self.entries)

    @property
    def d(self):
        """
        Returns the distance function
        """
        return self.mtree.d

    def is_full(self):
        """
        Check if the node is full (with M entries)
        """
        return len(self) == self.mtree.max_nodes

    def is_empty(self):
        return len(self) == 0

    def is_root(self):
        return self is self.mtree.root

    def remove_entry(self, entry):
        """Removes the entry from this node
        Raise KeyError if the entry is not in this node
        """
        self.entries.remove(entry)

    def add_entry(self, entry):
        """Add an entry to this node.
        Raise ValueError if the node is full.
        """
        if self.is_full():
            raise ValueError("On peut pas ajoutee l'entree %  en un noeud plein " % str(entry))
        self.entries.add(entry)

    #TODO recomputes d(leaf, parent)!
    def set_entries_and_parent_entry(self, new_entries, new_parent_entry):
        self.entries = new_entries
        self.parent_entry = new_parent_entry
        self.parent_entry.radius = self.covering_radius_for(self.parent_entry.obj)
        self._update_entries_distance_to_parent()


    #wastes d computations if parent hasn't changed.
    #How to avoid? -> check if the new routing_object is the same as the old
    # (compare id(obj) not obj directly to prevent == assumption about object?)
    def _update_entries_distance_to_parent(self):
        if self.parent_entry:
            for entry in self.entries:
                entry.distance_to_parent = self.d(entry.obj, self.parent_entry.obj)

    @abc.abstractmethod
    def add(self, obj):# pragma: no cover
        """Add obj into this subtree"""
        pass

    @abc.abstractmethod
    def covering_radius_for(self, obj):
        """Compute the radius needed for obj to cover the entries of this node.
        """
        pass

    @abc.abstractmethod
    def search(self, query_obj, pr, nn, d_parent_query):
        pass

    @abc.abstractmethod
    def rangeSearch(self, query_obj, pr, nn, d_parent_query, r):
        pass


#_____________________________________ Classe LeafNode __________________________________#
# Classe Represente un leaf noeud qui h'erite AbstractNode
class LeafNode(AbstractNode):
    """A leaf of the M-tree"""
    def __init__(self, mtree, parent_node=None, parent_entry=None, entries=None):
        AbstractNode.__init__(self, mtree, parent_node, parent_entry, entries)

    def add(self, obj):
        distance_to_parent = self.d(obj, self.parent_entry.obj) if self.parent_entry else None
        new_entry = MTEntry(obj, distance_to_parent)
        # if N is not full just insert the new object
        if not self.is_full():
            # Store entry(On) in N
            self.entries.add(new_entry)
        else:
            # The node is at full capacity, then it is needed to do a new split in this level
            # Split(N,entry(On));
            split(self, new_entry, self.d)

    def covering_radius_for(self, obj):
        """
            Compute minimal radius for obj so that it covers all the objects
            of this node.
        """
        # Calcule le rayon couvrant
        if not self.entries:
            return 0
        else:
            return max(map(lambda e: self.d(obj, e.obj), self.entries))

    # k_NN_InternalNode_Search _____________________________________________________

    def could_contain_results(self, query_obj, search_radius, distance_to_parent, d_parent_query):
        if self.is_root():
            return True
        return abs(d_parent_query - distance_to_parent) <= search_radius

    def search(self, query_obj, pr, nn, d_parent_query):
        """
        Executes the K-NN query.

        Arguments:
            query_obj: Q
            pr: PR
            nn: NN
            d_parent_query:
        """
        # for each entry(Oj) in N do:
        for entry in self.entries:
            if self.could_contain_results(query_obj, nn.search_radius(),entry.distance_to_parent,d_parent_query):

                distance_entry_to_q = self.d(entry.obj, query_obj)

                if distance_entry_to_q <= nn.search_radius():
                    if(type(entry.obj) == int):
                        nn.update(entry.obj, distance_entry_to_q)
                    else:
                        print("[INFO] Voila les distances ", distance_entry_to_q)
                        nn.update(list(entry.obj), distance_entry_to_q)

    def rangeSearch(self, query_obj, pr, nn, d_parent_query, r):
        """
        Executes the K-NN query.

        Arguments:
            query_obj: Q
            pr: PR
            nn: NN
            d_parent_query:
        """
        # for each entry(Oj) in N do:
        for entry in self.entries:
            if self.could_contain_results(query_obj, r,entry.distance_to_parent,d_parent_query):

                distance_entry_to_q = self.d(entry.obj, query_obj)

                if distance_entry_to_q <= r:
                    if(type(entry.obj) == int):
                        nn.update(entry.obj, distance_entry_to_q)
                    else:
                        nn.update(list(entry.obj), distance_entry_to_q)



#_____________________________________ Classe InternalNode __________________________________#

# Classe represent un noeud interne qui h'erite AbstractNode
class InternalNode(AbstractNode):
    """An internal node of the M-tree"""
    def __init__(self, mtree, parent_node=None, parent_entry=None, entries=None):
        AbstractNode.__init__(self, mtree, parent_node, parent_entry, entries)

    #TODO: apply optimization that uses the d of the parent to reduce the
    #number of d computation performed. cf M-Tree paper 3.3
    def add(self, obj):
        # put d(obj, e) in a dict to prevent recomputation
        # I guess memoization could be used to make code clearer but that is
        # too magic for me plus there is potentially a very large number of
        # calls to memoize
        dist_to_obj = {}
        for entry in self.entries:
            dist_to_obj[entry] = self.d(obj, entry.obj)

        def find_best_entry_requiring_no_covering_radius_increase():
            valid_entries = [e for e in self.entries if dist_to_obj[e] <= e.radius]
            return min(valid_entries, key=dist_to_obj.get) if valid_entries else None

        def find_best_entry_minimizing_radius_increase():
            entry = min(self.entries, key=lambda e: dist_to_obj[e] - e.radius)
            entry.radius = dist_to_obj[entry]
            return entry

        entry = find_best_entry_requiring_no_covering_radius_increase() or find_best_entry_minimizing_radius_increase()
        entry.subtree.add(obj)

    def covering_radius_for(self, obj):
        """Compute minimal radius for obj so that it covers the radiuses
        of all the routing objects of this node
        """
        if not self.entries:
            return 0
        else:
            return max(map(lambda e: self.d(obj, e.obj) + e.radius,self.entries))

    def set_entries_and_parent_entry(self, new_entries, new_parent_entry):
        AbstractNode.set_entries_and_parent_entry(self,new_entries,new_parent_entry)
        for entry in self.entries:
            entry.subtree.parent_node = self

    # k_NN_InternalNode_Search _____________________________________________________

    def could_contain_results(self, query_obj, search_radius, entry, d_parent_query):

        if self.is_root():
            return True

        return abs(d_parent_query - entry.distance_to_parent) <= search_radius + entry.radius

    def search(self, query_obj, pr, nn, d_parent_query):
        # for each entry(Or) in N do:
        for entry in self.entries:
            if self.could_contain_results(query_obj, nn.search_radius(), entry, d_parent_query):
                # Compute d(Or, Q);
                d_entry_query = self.d(entry.obj, query_obj)

                entry_dmin = max(d_entry_query - entry.radius, 0)

                if entry_dmin <= nn.search_radius():
                    heappush(pr, PrEntry(entry.subtree, entry_dmin, d_entry_query))
                    entry_dmax = d_entry_query + entry.radius
                    if entry_dmax < nn.search_radius():
                        nn.update(None, entry_dmax)


    def rangeSearch(self, query_obj, pr, nn, d_parent_query, r):
        # for each entry(Or) in N do:
        for entry in self.entries:
            if self.could_contain_results(query_obj, r, entry, d_parent_query):
                # Compute d(Or, Q);
                d_entry_query = self.d(entry.obj, query_obj)

                if d_entry_query <= r + entry.radius:
                    entry.subtree.rangeSearch(query_obj, pr, nn, d_parent_query, r)





#_____________________________________ SPLIT FUNCTION __________________________________#
# A lot of the code is duplicated to do the same operation on the existing_node
# and the new node :(. Could prevent that by creating a set of two elements and
# perform on the (two) elements of that set.
#TODO: Ugly, complex code. Move some code in Node/Entry?
def split(existing_node, entry, d):
    """
    Split existing_node N into two nodes N and N'.
    Adding entry to existing_node causes an overflow. Therefore we
    split existing_node into two nodes.

    Arguments:
        existing_node: N, full node to which entry should have been added
        entry: E, the added node. Caller must ensures that entry is initialized
               correctly as it would be if it were an effective entry of the node.
               This means that distance_to_parent must possess the appropriate
               value (the distance to existing_node.parent_entry).
        d: distance function.
    """
    # Get the reference to the M-Tree
    mtree = existing_node.mtree

    # The new routing objects are now all those in the node plus the new routing object
    all_entries = existing_node.entries | set((entry,)) # union


    # This part is already taken care of by the Entry class
    '''
        if N is not the root then
          {
             /*Get the parent node and the parent routing object*/
             let {\displaystyle O_{p}}O_{{p}} be the parent routing object of N
             let {\displaystyle N_{p}}N_{{p}} be the parent node of N
          }
    '''

    # This node will contain part of the objects of the node to be split */
    # Create a new node N'.
    # Type of the new node must be the same as existing_node
    # parent node, parent entry and entries are set later
    new_node = type(existing_node)(existing_node.mtree)

    #It is guaranteed that the current routing entry of the split node
    #(i.e. existing_node.parent_entry) is the one distance_to_parent
    #refers to in the entries (including the entry parameter).
    #Promote can therefore use distance_to_parent of the entries.

    # Promote two routing objects from the node to be split, to be new routing objects
    # Create new objects {\displaystyle O_{p1}}O_{{p1}} and {\displaystyle O_{p2}}O_{{p2}}.
    routing_object1, routing_object2 = mtree.promote(all_entries, existing_node.parent_entry, d)

    # Choose which objects from the node being split will act as new routing objects
    entries1, entries2 = mtree.partition(all_entries, routing_object1, routing_object2, d)

    message = "Error during split operation. All the entries have been assigned"
    message += "to one routing_objects and none to the other! Should never happen since at "
    message += "least the routing objects are assigned to their corresponding set of entries"
    assert entries1 and entries2, message


    # must save the old entry of the existing node because it will have
    # to be removed from the parent node later
    old_existing_node_parent_entry = existing_node.parent_entry

    # Setting a new parent entry for a node updates the distance_to_parent in
    # the entries of that node, hence requiring d calls.
    # promote/partition probably did similar d computations.
    # How to avoid recomputations between promote, partition and this?
    # share cache (a dict) passed between functions?
    # memoization? (with LRU!).
    #    id to order then put the two objs in a tuple (or rather when fetching
    #      try both way
    #    add a function to add value without computing them
    #      (to add distance_to_parent)

    #TODO: build_entry in the node method?
    # Store entries in each new routing object
    # Store in node N entries in N1 and in node N'entries in N2;
    existing_node_entry = MTEntry(routing_object1, None, None, existing_node)
    existing_node.set_entries_and_parent_entry(entries1, existing_node_entry)

    new_node_entry = MTEntry(routing_object2, None, None, new_node)
    new_node.set_entries_and_parent_entry(entries2, new_node_entry)


    if existing_node.is_root():
        # Create a new node and set it as new root and store the new routing objects Np
        new_root_node = InternalNode(existing_node.mtree)

        # Store Op1 and in Np
        existing_node.parent_node = new_root_node
        new_root_node.add_entry(existing_node_entry)
        # Store Op2 and in Np
        new_node.parent_node = new_root_node
        new_root_node.add_entry(new_node_entry)

        # Update the root
        mtree.root = new_root_node
    else:
        # Now use the parent routing object to store one of the new objects
        parent_node = existing_node.parent_node

        if not parent_node.is_root():
            # parent node has itself a parent, therefore the two entries we add
            # in the parent must have distance_to_parent set appropriately
            existing_node_entry.distance_to_parent = d(existing_node_entry.obj, parent_node.parent_entry.obj)
            new_node_entry.distance_to_parent = d(new_node_entry.obj, parent_node.parent_entry.obj)

        # Replace entry Op with entry Op1 in Np
        parent_node.remove_entry(old_existing_node_parent_entry)
        parent_node.add_entry(existing_node_entry)

        if parent_node.is_full():
            # If there is no free capacity then split the level up
            split(parent_node, new_node_entry, d)
        else:
            # The second routing object is stored in the parent
            # only if it has free capacity
            parent_node.add_entry(new_node_entry)
            new_node.parent_node = parent_node


# Pixel Info class (from sample code)
class ImageManager:
    def __init__(self, root, descriptor, distance, descDist, imgSize=(32, 32), imgFolder="default", imageFormat='.jpg', withIndexBase=False, radius=18):
        self.root = root
        self.imgSize = imgSize
        self.descriptor = descriptor
        self.distance = distance
        self.imgFolder = imgFolder
        self.descDist = descDist
        self.imgFormat = imageFormat
        self.radius = radius

        # CONSTANTS
        self.MOYENNE_STATISTIQUES = "Moyenne Statistiques"
        self.MOMENTS_STATISTIQUES = "Moments Statistiques"
        self.HISTOGRAMME_RGB = "Histogramme RGB"
        self.HISTOGRAMME_HSV = "Histogramme HSV"

        self.GABOR = "Filtre de Gabor"
        self.HARALICK = "Mesures de Haralick"

        self.MOMENTS_HU = "Moments de HU"
        self.MOMENTS_ZERNIKE = "Moments de Zernike"

        self.COLOR_TEXTURE = "Couleur et Texture"
        self.COLOR_SHAPE = "Couleur et Forme"

        ##############################
        ########### Mtree ############
        ##############################
        self.mtree = None
        if distance:
            self.mtree = MTree(distance, max_nodes=8)
        self.imageList = []
        self.photoList = []
        self.xmax = 125
        self.ymax = 132
        self.indexBase = []

        ##############################
        ####### Image Database #######
        ##############################
        # Add each image (for evaluation) into a list (from sample code)
        imgFolder = imgFolder+'/*' + self.imgFormat
        print("[DEBUG] Folder to be indexed: ", imgFolder)
        for infile in glob(imgFolder):
            im = PIL.Image.open(infile)
            # Resize the image for thumbnails.
            resized = im.resize((self.xmax, self.ymax), PIL.Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(resized)
            # Add the images to the lists.
            self.imageList.append(im)
            self.photoList.append(photo)
        if descriptor:
            # TODO: DONE  reading the saved descriptors
            # look for saved values
            #if os.path.isfile('indexBase.txt'):
            if withIndexBase:
                #print("[INFO] descDist[0]+'_indexBase/*.csv' = ", descDist[0]+'_indexBase/*.csv')
                #print("[INFO]-- Adding Images to the tree")
                data = []
                if descDist[0] == self.HISTOGRAMME_RGB: # Descriptor is Histogram
                    for index in glob(descDist[0]+'_indexBase/*.csv'):
                        data = readCSV_AVG(index)
                        # TODO: Add to M tree
                        csvHist = np.array(list(map(np.float32, data[1:])))
                        #print(csvHist)
                        hist = np.reshape(csvHist, (17, 17, 17))
                        self.addObjectsToTree([data[0], hist])
                        #print(".", end= " ")
                elif descDist[0] == self.HISTOGRAMME_HSV: # Descriptor is Histogram
                    for index in glob(descDist[0]+'_indexBase/*.csv'):
                        data = readCSV_AVG(index)
                        # TODO: Add to M tree
                        csvHist = list(map(np.float32, data[1:]))
                        self.addObjectsToTree([data[0], csvHist])
                        #print(".", end= " ")
                else:
                    for index in glob(descDist[0]+'_indexBase/*.csv'):
                        data = readCSV_AVG(index)
                        # TODO: Add to M tree
                        self.addObjectsToTree([data[0], data[1:]])
                        #print(".", end= " ")
                #print("\n[INFO]-- Insertion completed.")
                self.saveRawImagesFolder(data[0])
                messagebox.askyesno("L'indexation a réussi!", "Les signatures sont indexés en M-Tree.")
                

            # If not already computed, then compute the indexes
            else:
                # Compute
                #print("[INFO]-- Adding Images to the tree")
                if descDist[0] == self.MOYENNE_STATISTIQUES or\
                    descDist[0] == self.MOMENTS_STATISTIQUES or\
                        descDist[0] == self.HARALICK:
                    for im in self.imageList[:]:
                        # 1 get image data
                        fn, pixList = self.openImage(im)
                        # 2 get descriptor
                        avgs = [float(x) for x in descriptor(pixList)]
                        obj = [fn, avgs]
                        # TODO: 3 Add to M tree
                        self.addObjectsToTree(obj)
                        # 4 Save to desk
                        self.saveToDesk([im.filename, avgs])
                        #print(".", end= " ")
                elif descDist[0] == self.HISTOGRAMME_RGB: # Descriptor is Histogram
                    for im in self.imageList[:]:
                        # 1 get image data
                        fn, pixList = self.openImage(im)
                        # 2 get descriptor
                        hist = descriptor(pixList)
                        # TODO: 3 Add to M tree
                        obj = [fn, hist]
                        self.addObjectsToTree(obj)
                        # 4 Save to desk
                        self.saveToDesk([im.filename, hist.flatten()])
                        #print(".", end= " ")
                elif descDist[0] == self.HISTOGRAMME_HSV: # Descriptor is Histogram
                    for im in self.imageList[:]:
                        # 1 get image data
                        fn, pixList = self.openImage(im)
                        # 2 get descriptor
                        imData = cv2.imread(im.filename.replace("\\","/"))
                        imData = cv2.resize(imData, self.imgSize)
                        hist = descriptor(imData)
                        # TODO: 3 Add to M tree
                        obj = [fn, hist]
                        self.addObjectsToTree(obj)
                        # 4 Save to desk
                        self.saveToDesk([im.filename, hist])
                        #print(".", end= " ")
                elif descDist[0] == self.GABOR:
                    for im in self.imageList[:]:
                        # 1 get image data
                        fn = self.cleanFileName(im.filename)
                        #image  = cv2.imread(im.filename.replace("\\","/"), cv2.IMREAD_GRAYSCALE)
                        #imData = cv2.resize(image, self.imgSize)
                        # 2 get descriptor
                        img = im.filename.replace("\\","/")
                        avgs = [float(x) for x in descriptor(img)]
                        obj = [fn, avgs]
                        # TODO: 3 Add to M tree
                        self.addObjectsToTree(obj)
                        # 4 Save to desk
                        self.saveToDesk([im.filename, avgs])
                        #print(".", end= " ")
                elif descDist[0] == self.MOMENTS_HU:
                    for im in self.imageList[:]:
                        # 1 get image data
                        fn = self.cleanFileName(im.filename)
                        image  = cv2.imread(im.filename.replace("\\","/"), cv2.IMREAD_GRAYSCALE)
                        imData = cv2.resize(image, self.imgSize)
                        # 2 get descriptor
                        hu = [float(x) for x in descriptor(imData)]
                        obj = [fn, hu]
                        # TODO: 3 Add to M tree
                        self.addObjectsToTree(obj)
                        # 4 Save to desk
                        self.saveToDesk([im.filename, hu])
                        #print(".", end= " ")
                elif descDist[0] == self.MOMENTS_ZERNIKE:
                    for im in self.imageList[:]:
                        # 1 get image data
                        fn = self.cleanFileName(im.filename)
                        image  = cv2.imread(im.filename.replace("\\","/"), cv2.IMREAD_GRAYSCALE)
                        imData = cv2.resize(image, self.imgSize)
                        # 2 get descriptor
                        hu = [float(x) for x in descriptor(imData, self.radius)]
                        obj = [fn, hu]
                        # TODO: 3 Add to M tree
                        self.addObjectsToTree(obj)
                        # 4 Save to desk
                        self.saveToDesk([im.filename, hu])
                        #print(".", end= " ")
                elif descDist[0] == self.COLOR_TEXTURE:
                    for im in self.imageList[:]:
                        # 1 get image data
                        fn, pixList = self.openImage(im)
                        fn = self.cleanFileName(im.filename)
                        path  = im.filename.replace("\\","/")
                        # 2 get descriptor
                        hu = [float(x) for x in descriptor(pixList, path)]
                        obj = [fn, hu]
                        # TODO: 3 Add to M tree
                        self.addObjectsToTree(obj)
                        # 4 Save to desk
                        self.saveToDesk([im.filename, hu])
                        #print(".", end= " ")
                
                elif descDist[0] == self.COLOR_SHAPE:
                    for im in self.imageList[:]:
                        # 1 get image data
                        fn, pixList = self.openImage(im)
                        fn = self.cleanFileName(im.filename)
                        path  = im.filename.replace("\\","/")
                        image  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        gray = cv2.resize(image, self.imgSize)
                        # 2 get descriptor
                        hu = [float(x) for x in descriptor(pixList, gray)]
                        obj = [fn, hu]
                        # TODO: 3 Add to M tree
                        self.addObjectsToTree(obj)
                        # 4 Save to desk
                        self.saveToDesk([im.filename, hu])
                        #print(".", end= " ")
            
                #print("\n[INFO]-- Insertion completed.")
                messagebox.askyesno("Extraction des signatures a réussie!", "Les signatures sont stocké dans: "+descDist[0]+"_indexBase/*.csv")

    def openImage(self, im):
        fn = self.cleanFileName(im.filename)
        imData = cv2.imread(im.filename.replace("\\","/"))
        imData = cv2.resize(imData, self.imgSize)
        pixList = list(imData)
        return fn, pixList

    def saveToDesk(self, obj):
        fn = self.cleanFileName(obj[0])
        data = [obj[0]] # filepath: Absolute
        data.extend(obj[1]) # avgs, hist, moment, ...
        folderName = self.descDist[0]+'_indexBase/'
        if exists(folderName):
            writeCSV_AVG(folderName+fn+'.csv', data)
        else:
            mkdir(folderName)
            writeCSV_AVG(folderName+fn+'.csv', data)


    def cleanFileName(self, filename):
        fn = filename
        p = fn.rfind("\\")
        if p != -1:
            fn = fn[p+1:]
        else:
            p = fn.rfind("/")
            if p != -1:
                fn = fn[p+1:]
        return fn
    
    def saveRawImagesFolder(self, fileAbsolutePath):
        folder = fileAbsolutePath
        p = folder.rfind("\\")
        if p != -1:
            folder = folder[:p]
        else:
            p = folder.rfind("/")
            if p != -1:
                folder = folder = folder[:p]
        self.imgFolder = folder

    def addObjectsToTree(self, obj):
        self.mtree.add(obj)

    def executeImageSearch(self, feature, k=1):
        #print("[INFO]-- Executing Image Search with k = ",k)
        query = [None, feature]
        result_list = list(self.mtree.k_NN_search(query, k))
        #print("[INFO]-- Search finished")
        return result_list

    # Accessor functions
    def get_imageList(self):
        return self.imageList

    def get_photoList(self):
        return self.photoList

    def get_largePL(self):
        return self.largePL

    def get_xmax(self):
        return self.xmax

    def get_ymax(self):
        return self.ymax

    def getIndexBase(self):
        return self.indexBase




class CBIR_SIDI(Frame):
    def __init__(self, root, imageManager, w, h):
        self.w, self.h = w, h
        self.root = root
        self.basedOn = 1
        self.withIndexBase = False
        # variables
        self.imgManager = imageManager # image manager
        self.imageList = imageManager.get_imageList()
        self.photoList = imageManager.get_photoList()
        self.indexBase = imageManager.getIndexBase()
        self.imgSize = (32, 32)
        self.radius = IntVar()
        self.radius.set(22)

        self.xmax = imageManager.get_xmax() + 20
        self.ymax = imageManager.get_ymax() + 10
        self.currentPhotoList = imageManager.get_photoList()
        self.currentImageList = imageManager.get_imageList()

        # COLORS
        self.bgc = '#e8e8e8' # background
        self.btc = '#007EA4' # buttons
        self.abtc = '#F89D32' # active button
        self.fgc = '#ffffff' # forground
        self.basedOnC = ["#CF1F1F", "grey", "grey", "grey"]
        self.bth = 5 # button height
        self.currentPage = 0
        self.totalPages = self.get_totalPages()
        self.weights = []

        # CONSTANTS
        self.MOYENNE_STATISTIQUES = "Moyenne Statistiques"
        self.MOMENTS_STATISTIQUES = "Moments Statistiques"
        self.HISTOGRAMME_RGB = "Histogramme RGB"
        self.HISTOGRAMME_HSV = "Histogramme HSV"

        self.GABOR = "Filtre de Gabor"
        self.HARALICK = "Mesures de Haralick"

        self.MOMENTS_HU = "Moments de HU"
        self.MOMENTS_ZERNIKE = "Moments de Zernike"

        self.COLOR_TEXTURE = "Couleur et Texture"
        self.COLOR_SHAPE = "Couleur et Forme"

        self.EUCLIDIEN = "Euclidienne"
        self.MANHATAN = "Manhatan"
        self.CHISQRT = "CHI Square"
        self.BHATTACHARYYA = "Bhattacharyya"
        self.MANHATAN_FUSION = "Manhatan (Fusion)"
        self.EUCLID_FUSION = "Euclidienne (Fusion)"

        self.KVALUE = "Le nombre k : "


        # main frame
        #self.mainframe = Frame(root, bg=self.bgc, width=1366,  height=768)
        #self.mainframe.pack()

        self.mymenu = Menu(self.root, tearoff=False)
        self.root.config(menu=self.mymenu)
        self.file_menu = Menu(self.mymenu, tearoff=False)
        self.mymenu.add_cascade(label="Fichier", menu=self.file_menu)
        self.new_menu = Menu(self.file_menu, tearoff=False)
        self.file_menu.add_cascade(label="Choisir", menu=self.new_menu)
        self.new_menu.add_command(label="Base de données image/CSV ", command=self.browse_button)
        self.new_menu.add_command(label="Image requête", command=self.browse_buttonQ)
        self.file_menu.add_command(label="Ouvrir image", command=self.browse_buttonQ)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Enregistrer résultat", command=lambda: self.saveResults())
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Effacer résultat", command=lambda: self.deleteResults())
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Quitter", command=self.root.destroy)

        self.mymenu.add_command(label="Indexer", command=lambda: self.indexer())
        self.mymenu.add_command(label="Rechercher", command=lambda: self.find())

        self.view_menu = Menu(self.mymenu, tearoff=False)
        self.mymenu.add_cascade(label="Fenêtre", menu=self.view_menu)
        self.view_menu.add_command(label="Mesurer la qualité", command=lambda: self.mesurer())

        self.desc_menu = Menu(self.view_menu, tearoff=False)
        self.view_menu.add_cascade(label="Descripteurs", menu=self.desc_menu)
        self.desc_menu.add_command(label="Basé couleur", command=lambda: self.contentType(1))
        self.desc_menu.add_command(label="Basé texture", command=lambda: self.contentType(2))
        self.desc_menu.add_command(label="Basé forme", command=lambda: self.contentType(3))
        self.desc_menu.add_command(label="Composé", command=lambda: self.contentType(4))

        self.mymenu.add_command(label="Aide?", command=lambda: self.aide())
        self.mymenu.add_command(label="À propos!", command=lambda: self.aPropos())

        self.upperFrame = Frame(self.root)
        self.upperFrame.pack()

        self.btn_color = Button(self.upperFrame,
                                 text="Basé Couleur",
                                 font=('Arial',10,'bold'),
                                 width=41,
                                 pady=self.bth,
                                 border=5,
                                 bg=self.basedOnC[0],
                                 fg=self.fgc,
                                 activebackground=self.abtc,
                                 command=lambda: self.contentType(1))
        self.btn_color.grid(row=0, column=0)

        self.btn_texture = Button(self.upperFrame,
                                 text="Basé Texture",
                                 font=('Arial',10,'bold'),
                                 width=40,
                                 pady=self.bth,
                                 border=5,
                                 bg=self.basedOnC[1],
                                 fg=self.fgc,
                                 activebackground=self.abtc,
                                 command=lambda: self.contentType(2))
        self.btn_texture.grid(row=0, column=1)

        self.btn_shape = Button(self.upperFrame,
                                 text="Basé Forme",
                                 font=('Arial',10,'bold'),
                                 width=40,
                                 pady=self.bth,
                                 border=5,
                                 bg=self.basedOnC[2],
                                 fg=self.fgc,
                                 activebackground=self.abtc,
                                 command=lambda: self.contentType(3))
        self.btn_shape.grid(row=0, column=2)

        self.btn_fusion = Button(self.upperFrame,
                                 text="Fusion (Combinaison)",
                                 font=('Arial',10,'bold'),
                                 width=41,
                                 pady=self.bth,
                                 border=5,
                                 bg=self.basedOnC[3],
                                 fg=self.fgc,
                                 activebackground=self.abtc,
                                 command=lambda: self.contentType(4))
        self.btn_fusion.grid(row=0, column=3)

        self.lowerFrame = Frame(self.root, bg=self.bgc)
        self.lowerFrame.pack()

        #############################
        ####### Query Frame #########
        #############################


        self.queryFrame = LabelFrame(self.lowerFrame,
                                     bg=self.bgc,
                                     height=768,
                                     text="Section: Indexation et Recherche")
        self.queryFrame.grid(row=0, column=0)

        # control panel (Buttons)
        self.dbQueryPanel = Frame(self.queryFrame, bg=self.bgc)
        self.dbQueryPanel.pack()

        # Database control
        self.lbl_dbQueryPanel = Label(self.dbQueryPanel,
            text="OPTIONS: INDEXATION ET BASES DE DONNEES (OFFLINE)",
            fg="white", bg="gray",
            width=52,
            height=1,
            font=('Arial',10,'bold'))
        self.lbl_dbQueryPanel.pack(pady=2)

        self.var_choix = StringVar()
        self.canva_BDD = Canvas(self.dbQueryPanel, bg=self.bgc)
        self.canva_BDD.pack()
        self.rdio_ByImages = Radiobutton(self.canva_BDD,
                                        text="Dossier d'images",
                                        bg=self.bgc,
                                        variable=self.var_choix,
                                        value="Dossier d'images : ",
                                        width=25)

        self.rdio_ByCSVs = Radiobutton(self.canva_BDD,
                                        text="Dossier CSVs",
                                        bg=self.bgc,
                                        variable=self.var_choix,
                                        value="Dossier CSVs : ",
                                        width=25)
        self.var_choix.set("Dossier d'images : ")
        self.rdio_ByImages.grid(row=0, column=0)
        self.rdio_ByCSVs.grid(row=0, column=1)

        ######## Folder selection
        self.folder_path = StringVar()
        self.canva_folder = Canvas(self.dbQueryPanel, bg=self.bgc)
        self.canva_folder.pack()

        self.label_folder = Label(self.canva_folder,
                                  textvariable=self.var_choix,
                                  bg=self.bgc)
        self.label_folder.grid(row=0, column=0)
        self.entryFolder = Entry(self.canva_folder, width=40, textvariable=self.folder_path)
        self.entryFolder.grid(row=0, column=1)
        self.btn_browse = Button(self.canva_folder,
                                    text="Choisir",
                                    border=2,
                                    bg="#989ea6",
                                    fg=self.fgc,
                                    activebackground=self.abtc,
                                    command=self.browse_button)
        self.btn_browse.grid(row=0, column=2)

        #________________________________________________________
        self.lbl_distances = Label(self.dbQueryPanel,
            text="OPTIONS: CHOIX DE DESCRIPTEUR",
            fg="white", bg="gray",
            width=52, height=1,
            font=('Arial',10,'bold'))
        self.lbl_distances.pack(pady=2)

        self.offlineOptions = Canvas(self.dbQueryPanel, bg=self.bgc)
        self.offlineOptions.pack()
        ######## Descriptor selection
        self.var_desciptor = StringVar()
        optionList_desc = (self.MOYENNE_STATISTIQUES,
                           self.MOMENTS_STATISTIQUES,
                           self.HISTOGRAMME_RGB,
                           self.HISTOGRAMME_HSV)
        self.canva_desc = Canvas(self.offlineOptions, bg=self.bgc)
        self.canva_desc.grid(row=0, column=0)

        self.label_desc = Label(self.canva_desc,
                                bg=self.bgc,
                                text="Descripteur : ")
        self.label_desc.grid(row=0, column=0)
        self.var_desciptor.set(optionList_desc[0])
        self.om_descriptor = OptionMenu(self.canva_desc,
                            self.var_desciptor,
                             *optionList_desc
                    )
        self.om_descriptor.grid(row=0, column=1)

        #___________________________________________________________________________#
        self.imageOptions = Canvas(self.dbQueryPanel, bg=self.bgc)
        self.imageOptions.pack()
        ######## Type of image selection
        self.var_typeImg = StringVar()
        optionList_ti = ('.jpg', '.png')
        self.canva_typeImg = Canvas(self.imageOptions, bg=self.bgc)
        self.canva_typeImg.grid(row=0, column=1)

        self.label_typeImg = Label(self.canva_typeImg,
                                    text="Type d'images : ",
                                    bg=self.bgc)
        self.label_typeImg.grid(row=0, column=0)
        self.var_typeImg.set(optionList_ti[0])
        self.om_imgType = OptionMenu(self.canva_typeImg, self.var_typeImg, *optionList_ti)
        self.om_imgType.grid(row=0, column=1)

        self.imgSizeVar = IntVar()
        self.imgSizeVar.set(32)
        self.labelImgSize= Label(self.imageOptions,
                                    bg=self.bgc,
                                    text="Taille d'images N (NxN):")
        self.labelImgSize.grid(row=0, column=2)
        self.entryImgSize = Spinbox(self.imageOptions, width=4, from_ = 8, to = 80, textvariable=self.imgSizeVar)
        self.entryImgSize.grid(row=0, column=3)

        #__________________________________________________
        self.lbl_distances = Label(self.dbQueryPanel,
            text="OPTIONS: CHOIX DE LA MESURE DE SIMILARITE",
            fg="white", bg="gray",
            width=52, height=1,
            font=('Arial',10,'bold'))
        self.lbl_distances.pack(pady=2)

        ######## Distance selection
        self.var_distance = StringVar()
        optionListD = (self.MANHATAN,
                      self.EUCLIDIEN,
                       self.CHISQRT,
                       self.BHATTACHARYYA)
        self.canva_dist = Canvas(self.dbQueryPanel, bg=self.bgc)
        self.canva_dist.pack()

        self.label_dist = Label(self.canva_dist,
                                bg=self.bgc,
                                text="Choix du distance : ")
        self.label_dist.grid(row=0, column=0)

        self.var_distance.set(optionListD[0])
        self.omd = OptionMenu(self.canva_dist, self.var_distance, *optionListD)
        self.omd.grid(row=0, column=1)

        self.btn_indexer = Button(self.dbQueryPanel,
                                 text="Indexer",
                                 font=('Arial',10,'bold'),
                                 width=10,
                                 border=5,
                                 bg=self.btc,
                                 fg=self.fgc,
                                 activebackground=self.abtc,
                                 command=lambda: self.indexer())
        self.btn_indexer.pack()
        ######## Query image selection
        self.lbl_descriptrs = Label(self.dbQueryPanel,
            text="OPTIONS: CHOIX DE REQUETE (ONLINE)",
            fg="white", bg="gray",
            width=52, height=1,
            font=('Arial',10,'bold'))
        self.lbl_descriptrs.pack(pady=2)
        self.query_path = StringVar()
        self.canva_query = Canvas(self.dbQueryPanel, bg=self.bgc)
        self.canva_query.pack()

        self.label_query = Label(self.canva_query,
                                bg=self.bgc,
                                text="Image requête:")
        self.label_query.grid(row=0, column=0)
        self.entryQuery = Entry(self.canva_query, width=40, textvariable=self.query_path)
        self.entryQuery.grid(row=0, column=1)
        self.btn_browseQ = Button(self.canva_query,
                                    text="Choisir",
                                    border=2,
                                    bg="#989ea6",
                                    fg=self.fgc,
                                    activebackground=self.abtc,
                                    command=self.browse_buttonQ)
        self.btn_browseQ.grid(row=0, column=2)


        # selected image window
        self.selectedImage = Label(self.dbQueryPanel,
                                   width=320,
                                   height=200,
                                   bg=self.bgc)
        self.selectedImage.pack()
        self.update_preview(self.imageList[0].filename)
        #___________________________________________________________________________#

        ########### Buttond
        self.canva_btns = Canvas(self.dbQueryPanel)
        self.canva_btns.pack()
        self.btn_search = Button(self.canva_btns,
                                 text="Rechercher",
                                 font=('Arial',10,'bold'),
                                 width=15,
                                 pady=self.bth,
                                 border=5,
                                 bg=self.btc,
                                 fg=self.fgc,
                                 activebackground=self.abtc,
                                 command=lambda: self.find())
        self.btn_search.grid(row=0, column=0)

        self.btn_reset = Button(self.canva_btns,
                                text="Visualiser",
                                font=('Arial',10,'bold'),
                                width=15,
                                pady=self.bth,
                                border=5,
                                bg="#CF1F1F",
                                fg=self.fgc,
                                activebackground=self.abtc,
                                command=lambda: self.reset())
        self.btn_reset.grid(row=0, column=1)
        #___________________________________________________________________________#


        #__________________________________________________________________________________

        self.lbl_search = Label(self.dbQueryPanel,
            text="OPTIONS: RECHERCHE M-TREE",
            fg="white", bg="gray",
            width=52,
            height=1,
            font=('Arial',10,'bold'))
        self.lbl_search.pack(pady=2)

        self.var_searchMethod = StringVar()
        self.var_searchMethod.set(self.KVALUE)
        self.canva_SM = Canvas(self.dbQueryPanel, bg=self.bgc)
        self.canva_SM.pack()
        self.knn = Radiobutton(self.canva_SM,
                                text="k-NN",
                                bg=self.bgc,
                                variable=self.var_searchMethod,
                                value=self.KVALUE,
                                width =25)
        self.rquery = Radiobutton(self.canva_SM,
                                  text="Range query",
                                  bg=self.bgc,
                                  variable=self.var_searchMethod,
                                  value="Le rayon r : ",
                                  width =25)
        self.knn.grid(row=0, column=0)
        self.rquery.grid(row=0, column=1)

        self.KRange = IntVar()
        self.KRange.set(50)
        self.canva_KRange = Canvas(self.dbQueryPanel, bg=self.bgc)
        self.canva_KRange.pack()
        self.label_KRange = Label(self.canva_KRange,
                                    bg=self.bgc,
                                    textvariable=self.var_searchMethod)
        self.label_KRange.grid(row=0, column=0)
        self.entryKRange = Spinbox(self.canva_KRange, width=4,
                                    from_ = 0, to = 1000000,
                                    textvariable=self.KRange)
        self.entryKRange.grid(row=0, column=1)


        # _______________________________________________________________________ #
        ##################################
        # results frame
        ##################################

        self.resultsViewFrame = LabelFrame(self.lowerFrame,
                                           bg=self.bgc,
                                           text="Section: Résultats de recherche")
        self.resultsViewFrame.grid(row=0, column=1)

        instr = Label(self.resultsViewFrame,
                      bg=self.bgc,
                      fg='#aaa4bb',
                      text="Click l'image pour sélectionner.")
        instr.pack()

        self.resultPanel = LabelFrame(self.resultsViewFrame, bg=self.bgc)
        self.resultPanel.pack()
        self.canvas = Canvas(self.resultPanel ,
                             bg=self.bgc,
                             width=920,
                             height=520
                            )
        self.canvas.pack()

        # page navigation
        self.pageButtons = Frame(self.resultsViewFrame, bg=self.bgc)
        self.pageButtons.pack()

        self.btn_prev = Button(self.pageButtons, text="<< Page précédente",
                               width=30, border=5, bg=self.btc,
                               fg=self.fgc, activebackground=self.abtc,
                               command=lambda: self.prevPage())
        self.btn_prev.pack(side=LEFT)

        self.pageLabel = Label(self.pageButtons,
                               text="Page 1 de " + str(self.totalPages),
                               width=43, bg=self.bgc, fg='#aaaaaa')
        self.pageLabel.pack(side=LEFT)

        self.btn_next = Button(self.pageButtons, text="Page suivate>>",
                               width=30, border=5, bg=self.btc,
                               fg=self.fgc, activebackground=self.abtc,
                               command=lambda: self.nextPage())
        self.btn_next.pack(side=RIGHT)


    def mesurer(self):
        self.mesurUI = Toplevel(self.root)
        self.mesurUI.geometry("760x600")
        self.mesurUI.title("Mesure de la pertinence de la requête.")
        self.mesurUI.iconbitmap("sidicbir.ico")

        self.width_titre = 75
        self.bg_titre = "#4ECDC4"
        self.bg_tab = "white"
        self.bg_resoudre = 'green'
        self.bg_effacer = '#FF6B6B'
        self.fg_text = "green"
        self.topsection = Canvas(self.mesurUI, bg=self.bg_titre)
        self.topsection.pack(pady=10)

        self.titre = Label(self.topsection)
        self.titre.grid(row=0, column=0)
        self.topLabel = Label(self.titre,
                              text="MESURE DE PERTINENCE D'UNE REQUETE DE RECHERCHE D'IMAGE PAR CONTENU",
                              bg=self.bg_titre,
                              font=('Arial',12,'bold'),
                              width=self.width_titre)
        self.topLabel.pack()
        self.topLabel1 = Label(self.titre,
                              text="AVEC LE SYSTEM SIDICBIR",
                              bg=self.bg_titre,
                              font=('Arial',12,'bold'),
                              width=self.width_titre)
        self.topLabel1.pack()
        self.topLabel1 = Label(self.titre,
                              text="-v.2020.01-",
                              bg=self.bg_titre,
                              font=('Arial',12,'bold'),
                              width=self.width_titre)
        self.topLabel1.pack()



        self.baseCanva = Canvas(self.mesurUI)
        self.baseCanva.pack(pady=5)
        self.label_base = Label(self.baseCanva,
                                bg=self.bgc,
                                font=('Arial',8,'bold'),
                                text="Dossier de la base d'image utilsée:")
        self.label_base.grid(row=0, column=0)

        self.entryBase = Entry(self.baseCanva, width=75, textvariable=self.folder_path)
        self.entryBase.grid(row=0, column=1)

        self.classCanva = Canvas(self.mesurUI)
        self.classCanva.pack(pady=5)
        self.label_queryClass = Label(self.classCanva,
                                bg=self.bgc,
                                font=('Arial',12,'bold'),
                                text="Classe d'image requête:")
        self.label_queryClass.grid(row=0, column=0)

        self.classe = StringVar()
        self.classe.set(self.imgManager.cleanFileName(self.selected.filename))
        self.entryQueryClass = Entry(self.classCanva, width=20, textvariable=self.classe)
        self.entryQueryClass.grid(row=0, column=1)
        self.btn_browseQC = Button(self.mesurUI, text="Afficher la matrice de confusion",
                                    font=('Arial',10,'bold'),
                                    width=40,
                                    pady=self.bth,
                                    border=5,
                                    bg=self.btc,
                                    fg=self.fgc,
                                    activebackground=self.abtc,
                                    command=lambda: self.confusionMatrix())
        self.btn_browseQC.pack()

        self.CMcanva = Canvas(self.mesurUI)
        self.CMcanva.pack(pady=5)
        self.ResCanva = Canvas(self.mesurUI)
        self.ResCanva.pack(pady=5)

        self.mesurUI.mainloop()

    def confusionMatrix(self):
        self.ResCanva.destroy()
        self.CMcanva.destroy()

        listAll = []
        if self.withIndexBase:
            listAll = glob(self.imgManager.imgFolder+'/*'+self.imgManager.imgFormat)
        else:
            listAll = glob(self.folder_path.get()+'/*'+self.imgManager.imgFormat)
        true = []
        first, all = -1, 0
        for i in range(len(listAll)):
            if self.classe.get() in listAll[i]:
                true.append(1)
                if first == -1: first = i
                all += 1
            else:
                true.append(0)
        pred = true.copy()
        item = ""
        j, tmp, found = first, 0, 0
        for i in range(len(self.results)):
            item = self.results[i]
            if item != 'None':
                #print("[DEBUG] ", self.classe.get(), item)
                if self.withIndexBase:
                    item = self.imgManager.cleanFileName(item)
                if self.classe.get() in item:
                    found += 1
                else:
                    tmp += 1

        for i in range(len(pred)):
            if  not(first <= i <= first+all-1) and tmp > 0:
                pred[i] = 1
                tmp -= 1
        notfound = all - found
        #print("notfound ", notfound, first, all)
        for i in range(len(pred)):
            if  first <= i <= first+all-1 and notfound > 0 and pred[i]==1:
                pred[i] = 0
                notfound -= 1
        cm = confusion_matrix(true, pred)
        tn, fp, fn, tp = cm.ravel()
        #print("(tn, fp, fn, tp) = ", (tn, fp, fn, tp))

        #_______________________________________
        # TODO: do it with some clean code
        self.CMcanva = Canvas(self.mesurUI)
        self.CMcanva.pack()
        view = self.CMcanva

        self.positif = IntVar()
        self.positif.set(10)

        self.canvaPositif = Canvas(view, bg=self.bgc)
        self.canvaPositif.pack(pady=5)
        self.labelPositif = Label(self.canvaPositif,
                                    bg=self.bgc,
                                    font=('Arial',8,'bold'),
                                    text="Nombre total d'images pertinentes:")
        self.labelPositif.grid(row=0, column=0)
        self.entryPositif = Spinbox(self.canvaPositif, width=4, from_ = 0, to = 100,

                                    textvariable=self.positif)
        self.entryPositif.grid(row=0, column=1)

        self.vp = IntVar()
        self.vp.set(10)
        self.VPCanva = Canvas(view, bg=self.bgc)
        self.VPCanva.pack()
        self.VPLabel = Label(self.VPCanva,
                                    bg=self.bgc,
                                    font=('Arial',8,'bold'),
                                    text="Nombre d'images pertinentes retrouvées:")
        self.VPLabel.grid(row=0, column=0)
        self.VPEntry = Spinbox(self.VPCanva, width=4, from_ = 0, to = 100,
                                textvariable=self.vp)
        self.VPEntry.grid(row=0, column=1)

        self.found = IntVar()
        try:
            self.found.set(self.resultsLenght)
        except AttributeError:
            pass
        self.foundCanva = Canvas(view, bg=self.bgc)
        self.foundCanva.pack()
        self.foundLabel = Label(self.foundCanva,
                                    bg=self.bgc,
                                    font=('Arial',8,'bold'),
                                    text="Nombre total d'images retrouvées:")
        self.foundLabel.grid(row=0, column=0)
        self.foundEntry = Spinbox(self.foundCanva, width=4,
                               from_ = 0, to = 100,
                               textvariable=self.found)
        self.foundEntry.grid(row=0, column=1)
        #________________________
        self.found.set(tp+fp)
        self.vp.set(tp)
        self.positif.set(tp+fn)
        #print("Rappel :", tp/(tp+fn)) print("Precision :", tp/(tp+fp))

        self.posNegCanva = Canvas(view)
        self.posNegCanva.pack()
        self.label= Label(self.posNegCanva, width=20)
        self.label.grid(row=0, column=0)
        self.label_positif = Label(self.posNegCanva,
                                bg="red",
                                width=20,
                                font=('Arial',10,'bold'),
                                text="Positif",
                                border=1,
                                pady=1,
                                padx=2
                                )
        self.label_positif.grid(row=0, column=1)
        self.label_negatif = Label(self.posNegCanva,
                                bg="red",
                                width=20,
                                font=('Arial',10,'bold'),
                                text="Négatif",
                                border=1,
                                pady=1,
                                padx=2)
        self.label_negatif.grid(row=0, column=2)

        self.trueCanva = Canvas(view)
        self.trueCanva.pack()
        self.label_true= Label(self.trueCanva,
                                bg="green",
                                width=18,
                                font=('Arial',10,'bold'),
                                text=" Vrai ",
                                border=1,
                                pady=1,
                                )
        self.label_true.grid(row=0, column=0)

        self.entryTP = Label(self.trueCanva, width=23,
                                        text=str(tp),
                                        bg="#8aed8a")
        self.entryTP.grid(row=0, column=1)

        self.entryTN = Label(self.trueCanva, width=23,
                                        text=str(tn),
                                        bg="#d7fce4")
        self.entryTN.grid(row=0, column=2)

        self.falseCanva = Canvas(view)
        self.falseCanva.pack()
        self.label_false= Label(self.falseCanva,
                                bg="green",
                                width=18,
                                font=('Arial',10,'bold'),
                                text="Faux",
                                border=1,
                                pady=1,
                                )
        self.label_false.grid(row=0, column=0)

        self.entryFP = Label(self.falseCanva,
                                width=23,
                                text=str(fp),
                                bg="#f78f95")
        self.entryFP.grid(row=0, column=1)

        self.entryFN = Label(self.falseCanva, width=23,
                                        text=str(fn),
                                        bg="#7b9c5a")
        self.entryFN.grid(row=0, column=2)

        self.btn_calcul = Button(view,
                                 text="Mesurer la pertinence",
                                 font=('Arial',10,'bold'),
                                 width=40,
                                 pady=self.bth,
                                 border=5,
                                 bg=self.btc,
                                 fg=self.fgc,
                                 activebackground=self.abtc,
                                 command=lambda: self.getQualityMeasures())
        self.btn_calcul.pack(pady=5)









    def getQualityMeasures(self):
        try:
            self.ResCanva.destroy()
        except AttributeError:
            pass

        self.ResCanva = Canvas(self.mesurUI)
        self.ResCanva.pack()
        view = self.ResCanva
        recall = self.vp.get() / self.positif.get()
        precision = self.vp.get() / self.found.get()
        F_score = 2*recall*precision /(recall+precision)

        self.table = Canvas(view)
        self.table.pack(pady=10)
        lst = [["Rappel", "Précision", "F-mesure"], list(map(lambda x: round(x, 4), [recall, precision, F_score]))]
        self.Table(self.table, lst , bg="white")


    def Table(self, canva, lst, bg):
        # code for creating table
        total_rows = 2
        total_columns = 3
        self.e = ""
        for i in range(total_rows):
            for j in range(total_columns):
                if i == 0:
                    self.e = Entry(canva, width=16,
                                    fg='white',
                                    font=('Arial',10,'bold'),
                                    justify='center',
                                    bg="gray")
                else:
                    self.e = Entry(canva, width=16,
                                    fg='blue',
                                    font=('Arial',10,'bold'),
                                    justify='center',bg=bg)
                self.e.grid(row=i, column=j)
                self.e.insert(END, lst[i][j])








    def contentType(self, code):
        self.basedOnC[code-1], self.basedOnC[self.basedOn-1] = self.basedOnC[self.basedOn-1], self.basedOnC[code-1]
        self.btn_color.config(bg=self.basedOnC[0])
        self.btn_texture.config(bg=self.basedOnC[1])
        self.btn_shape.config(bg=self.basedOnC[2])
        self.btn_fusion.config(bg=self.basedOnC[3])

        self.basedOn = code
        self.change()

    def browse_button(self):
        # Allow user to select a directory and store it in global var
        # called folder_path
        filename = filedialog.askdirectory()
        self.folder_path.set(filename)

    def browse_buttonQ(self):
        # Allow user to select a file and store it in global var
        # called query_path
        filename = filedialog.askopenfilename()
        self.query_path.set(filename)
        self.update_preview(filename)

    def indexer(self):
        #TODO: Add pop up window
        if self.folder_path.get() == "":
            messagebox.showwarning("Message d'alert!", "Veuillez choisir un dossier SVP!")
            #print("[INFO] SELECTED FOLDER: empty path")


        # **************** CHOIX: DESCRIPTOR n DISTANCE *****************
        colorDescriptor = ColorDescriptor()
        textureDescriptor = TextureDescriptor()
        shapeDescriptor = ShapeDescriptor()
        distance = Distance()

        DESC = colorDescriptor.getAvgs
        DIST = distance.euclid
        descDist = [self.MOYENNE_STATISTIQUES, self.EUCLIDIEN]

        if self.var_desciptor.get() == self.MOMENTS_STATISTIQUES:
            DESC = colorDescriptor.getMoments
            descDist[0] = self.MOMENTS_STATISTIQUES
        elif self.var_desciptor.get() == self.HISTOGRAMME_RGB:
            DESC = colorDescriptor.getHist
            descDist[0] = self.HISTOGRAMME_RGB
        elif self.var_desciptor.get() == self.HISTOGRAMME_HSV:
            DESC = colorDescriptor.getHistHSV
            descDist[0] = self.HISTOGRAMME_HSV
        elif self.var_desciptor.get() == self.MOYENNE_STATISTIQUES:
            DESC = colorDescriptor.getAvgs
            descDist[0] = self.MOYENNE_STATISTIQUES
        elif self.var_desciptor.get() == self.GABOR:
            #DESC = textureDescriptor.getGaborFeatures
            #descDist[0] = self.GABOR
            g = Gabor()
            DESC = g.gabor_histogram
            descDist[0] = self.GABOR
        elif self.var_desciptor.get() == self.HARALICK:
            DESC = textureDescriptor.getHaralickFeatures
            descDist[0] = self.HARALICK
        elif self.var_desciptor.get() == self.MOMENTS_HU:
            DESC = shapeDescriptor.getHuMoments
            descDist[0] = self.MOMENTS_HU
        elif self.var_desciptor.get() == self.MOMENTS_ZERNIKE:
            DESC = shapeDescriptor.getZernikeMoments
            descDist[0] = self.MOMENTS_ZERNIKE
        elif self.var_desciptor.get() == self.COLOR_TEXTURE:
            fusionDescriptors = FusionDescriptors(self.w1.get(), self.w2.get())
            DESC = fusionDescriptors.getMomentsAndGabor
            descDist[0] = self.COLOR_TEXTURE
        elif self.var_desciptor.get() == self.COLOR_SHAPE:
            fusionDescriptors = FusionDescriptors(self.w1.get(), self.w2.get())
            DESC = fusionDescriptors.getMomentsAndZernike
            descDist[0] = self.COLOR_SHAPE


        if self.var_distance.get() == self.EUCLIDIEN:
            DIST = distance.euclid
            descDist[1] = self.EUCLIDIEN
        elif self.var_distance.get() == self.CHISQRT:
            DIST = distance.chi
            descDist[1] = self.CHISQRT
        elif self.var_distance.get() == self.BHATTACHARYYA:
            DIST = distance.Bhattachatyya
            descDist[1] = self.BHATTACHARYYA
        elif self.var_distance.get() == self.MANHATAN:
            DIST = distance.manhatan
            descDist[1] = self.MANHATAN
        elif self.var_distance.get() == self.MANHATAN_FUSION:
            DIST = fusionDescriptors.manhatanDistance
            descDist[1] = self.MANHATAN_FUSION

        print("[INFO] DESC = ", descDist[0])
        print("[INFO] DIST = ", descDist[1])

        # TODO: Save Images/Indexes database related folder
        imgFolder = self.folder_path.get()
        if self.var_choix.get() == "Dossier CSVs : ":
            self.withIndexBase = True
            # Get the folder path from a csv file
            csvFile = glob(imgFolder+'/*.csv')[0]
            data = readCSV_AVG(csvFile)
            self.imgManager.saveRawImagesFolder(data[0])
            imgFolder = self.imgManager.imgFolder
            print("[DEBUG] Raw Images Folder", imgFolder)

        imageFormat = self.var_typeImg.get()
        self.imgSize = (self.imgSizeVar.get(), self.imgSizeVar.get())

        if descDist[1] == self.BHATTACHARYYA or descDist[1] == self.CHISQRT and descDist[0] != self.HISTOGRAMME_RGB:
            messagebox.showwarning("Message d'alert", "Les distances Chi-Square et Bhachattaryya vont avec l'histogramme RGB!")
        else:
            self.imgManager = ImageManager(self.root, DESC, DIST, descDist, self.imgSize, imgFolder, imageFormat, self.withIndexBase, self.radius.get())

            self.imageList = self.imgManager.get_imageList()
            self.photoList = self.imgManager.get_photoList()
            self.indexBase = self.imgManager.getIndexBase()

            self.xmax = self.imgManager.get_xmax() + 20
            self.ymax = self.imgManager.get_ymax() + 10
            self.currentPhotoList = self.imgManager.get_photoList()
            self.currentImageList = self.imgManager.get_imageList()
            self.totalPages = self.get_totalPages()

    def find(self):
        queryFeature = []
        if self.imgManager.descDist[0] == self.MOMENTS_ZERNIKE:
            # 1 get image data
            image  = cv2.imread(self.selected.filename.replace("\\","/"), cv2.IMREAD_GRAYSCALE)
            imData = cv2.resize(image, self.imgSize)
            # 2 get descriptor
            queryFeature = [float(x) for x in self.imgManager.descriptor(imData, self.radius.get())]
        elif self.imgManager.descDist[0] == self.COLOR_TEXTURE:
            # 1 get image data
            fn, pixList = self.imgManager.openImage(self.selected)
            path = self.selected.filename.replace("\\","/")
            # 2 get descriptor
            queryFeature  = [float(x) for x in self.imgManager.descriptor(pixList, path)]
        elif self.imgManager.descDist[0] == self.COLOR_SHAPE:
            # 1 get image data
            fn, pixList = self.imgManager.openImage(self.selected)
            path = self.selected.filename.replace("\\","/")
            image  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            gray = cv2.resize(image, self.imgSize)
            # 2 get descriptor
            queryFeature  = [float(x) for x in self.imgManager.descriptor(pixList, gray)]
        elif self.imgManager.descDist[0] == self.MOMENTS_HU:
            # 1 get image data
            image  = cv2.imread(self.selected.filename.replace("\\","/"), cv2.IMREAD_GRAYSCALE)
            imData = cv2.resize(image, self.imgSize)
            # 2 get descriptor
            queryFeature = [float(x) for x in self.imgManager.descriptor(imData)]
        elif self.imgManager.descDist[0] == self.GABOR:
            # 1 get image data
            img = self.selected.filename.replace("\\","/")
            # 2 get descriptor
            queryFeature = [float(x) for x in self.imgManager.descriptor(img)]
        else:
            im = cv2.imread(self.selected.filename)
            im = cv2.resize(im, self.imgSize)
            try:
                queryFeature = self.imgManager.descriptor(im)
            except TypeError:
                messagebox.showwarning("Message d'alert", "Veuillez séléctionner/choisir une image SVP")

        if self.var_searchMethod.get() == self.KVALUE:
            self.results = self.imgManager.executeImageSearch(queryFeature, self.KRange.get())
        else:
            self.results = self.imgManager.executeImageRSearch(queryFeature, self.KRange.get())
        self.currentImageList, self.currentPhotoList = [], []
        self.resultsLenght = 0
        self.tempList = []
        for img in self.results:
            if img != 'None':
                self.resultsLenght += 1
                if self.withIndexBase:
                    im = PIL.Image.open(img) #.replace("/", "")
                    self.tempList.append(self.imgManager.cleanFileName(img))
                else:
                    im = PIL.Image.open(self.imgManager.imgFolder + "/" + img) #.replace("/", "")
                    self.tempList.append(img)
                # Resize the image for thumbnails.
                resized = im.resize((128, 128), PIL.Image.ANTIALIAS)
                photo = ImageTk.PhotoImage(resized)
                self.currentImageList.append(im)
                self.currentPhotoList.append(photo)


        iL = self.currentImageList[:24]
        pL = self.currentPhotoList[:24]
        self.currentPage = 0
        self.update_results((iL, pL))

    def destroyFusionOptions(self):
        try:
            self.label_cw.destroy()
            self.entryCW.destroy()
            self.label_tosw.destroy()
            self.entryToSW.destroy()
        except AttributeError:
            pass
    def destroyZernikeOptions(self):
        try:
            self.labelRadius.destroy()
            self.entryRadius.destroy()
        except AttributeError:
            pass
    def change(self):
        """
        Resets the GUI to its initial state
        """
        # initial display photos
        if self.basedOn == 1:
            optionList_desc = (self.MOYENNE_STATISTIQUES,
                           self.MOMENTS_STATISTIQUES,
                           self.HISTOGRAMME_RGB,
                           self.HISTOGRAMME_HSV)
            optionListD = (self.MANHATAN,
                      self.EUCLIDIEN,
                       self.CHISQRT,
                       self.BHATTACHARYYA)
            self.destroyFusionOptions()
            self.destroyZernikeOptions()
        elif self.basedOn == 2:
            optionList_desc = (self.GABOR, self.HARALICK)
            optionListD = (self.MANHATAN,
                      self.EUCLIDIEN)
            self.destroyFusionOptions()
            self.destroyZernikeOptions()
        elif self.basedOn == 3:
            optionList_desc = (self.MOMENTS_HU, self.MOMENTS_ZERNIKE)
            optionListD = (self.MANHATAN,
                      self.EUCLIDIEN)
            self.destroyFusionOptions()

            self.labelRadius= Label(self.canva_desc,
                                        bg=self.bgc,
                                        text="Radius (Zernike):")
            self.labelRadius.grid(row=0, column=2)
            self.entryRadius = Spinbox(self.canva_desc, width=4,
                                        from_ = 8, to = 80,
                                        textvariable=self.radius)
            self.entryRadius.grid(row=0, column=3)
        elif self.basedOn == 4:
            self.w1 = DoubleVar()
            self.w1.set(0.5)
            self.label_cw = Label(self.canva_dist,
                                        bg=self.bgc,
                                        text="W1")
            self.label_cw.grid(row=0, column=2)
            self.entryCW = Spinbox(self.canva_dist,
                                    width=4,
                                    from_ = 0, to = 1,
                                    textvariable=self.w1)
            self.entryCW.grid(row=0, column=3)

            self.w2 = DoubleVar()
            self.w2.set(0.5)
            self.label_tosw = Label(self.canva_dist,
                                        bg=self.bgc,
                                        text="W2")
            self.label_tosw.grid(row=0, column=4)
            self.entryToSW = Spinbox(self.canva_dist,
                                    width=4,
                                    from_ = 0, to = 1,
                                    textvariable=self.w2)
            self.entryToSW.grid(row=0, column=5)

            optionList_desc = (self.COLOR_TEXTURE, self.COLOR_SHAPE)
            optionListD = (self.MANHATAN_FUSION, self.EUCLID_FUSION)
            self.destroyZernikeOptions()


        self.var_desciptor.set(optionList_desc[0])
        self.om_descriptor.destroy()

        self.om_descriptor = OptionMenu(self.canva_desc,
                                        self.var_desciptor,
                                        *optionList_desc)
        self.om_descriptor.grid(row=0, column=1)

        self.omd.destroy()
        self.var_distance.set(optionListD[0])
        self.omd = OptionMenu(self.canva_dist,
                              self.var_distance,
                              *optionListD)
        self.omd.grid(row=0, column=1)

        self.deleteResults()


    def deleteResults(self):
        self.canvas.destroy()
        self.canvas = Canvas(self.resultPanel ,
                             bg=self.bgc,
                             width=920,
                             height=520
                            )
        self.canvas.pack()

    def reset(self):
        self.update_preview(self.imageList[0].filename)
        self.currentImageList = self.imageList
        self.currentPhotoList = self.photoList
        il = self.currentImageList[:24]
        pl = self.currentPhotoList[:24]
        self.update_results((il, pl))


    def get_pos(self, filename):
        """
        find selected image position
        :param filename:
        """
        pos = -1
        for i in range(len(self.imageList)):
            f = self.imageList[i].filename.replace("\\", "/")
            if filename == f:
                pos = i
        return pos


    def update_results(self, st):
        """
        Updates the photos in results window (used from sample)
        :param st: (image, photo)
        """

        txt = "Page "+str(self.currentPage + 1)+" of "+str(self.totalPages)
        self.pageLabel.configure(text=txt)
        cols = 6 # number of columns
        self.canvas.delete(ALL)


        #################################################
        self.canvas.config(width=920, height=550)
        self.canvas.pack()

        photoRemain = [] # photos to show
        for i in range(len(st[0])):
            f = st[0][i].filename
            img = st[1][i]
            photoRemain.append((f, img))

        rowPos = 0
        while photoRemain:
            photoRow = photoRemain[:cols]
            photoRemain = photoRemain[cols:]
            colPos = 0
            # Put a row
            for (filename, img) in photoRow:
                imagesFrame = Frame(self.canvas, bg=self.bgc, border=0)
                imagesFrame.pack(padx = 15)

                fn = self.imgManager.cleanFileName(filename)
                lbl = Label(imagesFrame,text=fn)
                lbl.pack()
                # Put image as a button
                handler = lambda f=filename: self.update_preview(f)
                btn_image = Button(imagesFrame,
                                   image=img,
                                   border=4,
                                   bg=self.bgc,
                                   width= 128,#self.imgManager.get_xmax()+20,
                                   activebackground=self.bgc
                                   )
                btn_image.config(command=handler)
                btn_image.pack(side=LEFT, padx=15)

                self.canvas.create_window(colPos,
                                          rowPos,
                                          anchor=NW,
                                          window=imagesFrame,
                                          width=self.xmax,
                                          height=self.ymax)
                colPos += self.xmax
            rowPos += self.ymax

    def update_preview(self, f):
        """
        Updates the selected images window. Image show!
        :param f: image identifier (the first image by default)
        """
        self.selected = PIL.Image.open(f.replace("\\", "/"))
        resized = self.selected.resize((320, 200), PIL.Image.ANTIALIAS)
        self.selectedPhoto = ImageTk.PhotoImage(resized)
        self.selectedImage.configure(image=self.selectedPhoto)

    # updates results page to previous page
    def prevPage(self):
        self.currentPage -= 1
        if self.currentPage < 0:
            self.currentPage = self.totalPages - 1
        start = self.currentPage * 24
        end = start + 24
        iL = self.currentImageList[start:end]
        pL = self.currentPhotoList[start:end]
        self.update_results((iL, pL))

    # updates results page to next page
    def nextPage(self):
        self.currentPage += 1
        if self.currentPage >= self.totalPages:
            self.currentPage = 0
        start = self.currentPage * 24
        end = start + 24
        iL = self.currentImageList[start:end]
        pL = self.currentPhotoList[start:end]
        self.update_results((iL, pL))

    # computes total pages in results
    def get_totalPages(self):
        pages = len(self.photoList) // 24
        if len(self.photoList) % 24 > 0:
            pages += 1
        return pages

    def saveResults(self):
        dirName = filedialog.askdirectory()
        for im, name in zip(self.currentImageList, self.tempList):
            im.save(dirName + "\\" + name)

    def aPropos(self):
        self.apUI = Toplevel(self.root)
        self.apUI.geometry("760x130")
        self.apUI.title("À propos!")
        self.apUI.iconbitmap("sidicbir.ico")

        self.width_titre = 75
        self.bg_titre = "#4ECDC4"
        self.bg_tab = "white"
        self.bg_resoudre = 'green'
        self.bg_effacer = '#FF6B6B'
        self.fg_text = "green"
        self.topsectionap = Canvas(self.apUI, bg=self.bg_titre)
        self.topsectionap.pack(pady=10)

        self.titreap = Label(self.topsectionap)
        self.titreap.grid(row=0, column=0)
        self.topLabelap = Label(self.titreap,
                              text="APPLICATION DE RECHERCHE D'IMAGE PAR CONTENU: SIDICBIR",
                              bg=self.bg_titre,
                              font=('Arial',12,'bold'),
                              width=self.width_titre)
        self.topLabelap.pack()
        self.topLabel1ap = Label(self.titreap,
                              text="MASTER SIDI, FST ERRACHIDIA",
                              bg=self.bg_titre,
                              font=('Arial',12,'bold'),
                              width=self.width_titre)
        self.topLabel1ap.pack()
        self.topLabel1ap = Label(self.titreap,
                              text="-v.2020.01-",
                              bg=self.bg_titre,
                              font=('Arial',12,'bold'),
                              width=self.width_titre)
        self.topLabel1ap.pack()
        self.topLabel2ap = Label(self.titreap,
                              text="Développé par: Hassan ZEKKOURI",
                              bg=self.bg_titre,
                              font=('Arial',12,'bold'),
                              width=self.width_titre)
        self.topLabel2ap.pack()

    def aide(self):
        self.apUI = Toplevel(self.root)
        self.apUI.geometry("760x50")
        self.apUI.title("Aide?")
        self.apUI.iconbitmap("sidicbir.ico")

        self.width_titre = 75
        self.bg_titre = "#4ECDC4"
        self.bg_tab = "white"
        self.bg_resoudre = 'green'
        self.bg_effacer = '#FF6B6B'
        self.fg_text = "green"
        self.topsectionap = Canvas(self.apUI, bg=self.bg_titre)
        self.topsectionap.pack(pady=10)

        self.titreap = Label(self.topsectionap)
        self.titreap.grid(row=0, column=0)
        self.topLabelap = Label(self.titreap,
                              text="Consulter: https://github.com/DgrinderHZ/SIDICBIR",
                              bg=self.bg_titre,
                              font=('Arial',12,'bold'),
                              width=self.width_titre)
        self.topLabelap.pack()



if __name__ == '__main__':
    root = Tk()
    # root.resizable(widthFalse, height=False)
    root.iconbitmap("sidicbir.ico")
    root.title("Système de recherche d'image par le conrnu - CBIR v.2020.01")
    #root.attributes('-fullscreen', True)
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry("%dx%d" % (w, h))
    descDist = ["Moyenne Statistiques", "Euclidienne"]
    root.configure(bg='#e8e8e8')
    pix = ImageManager(root, None, None, descDist)
    top = CBIR_SIDI(root, pix, w, h)
    root.mainloop()
