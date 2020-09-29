
import  os
from glob import glob
from PIL import ImageTk, Image
from mtree import *
import cv2
import csvmanager
import numpy as np
from os.path import exists
from os import mkdir

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
        self.xmax = 0
        self.ymax = 0
        self.x = 0
        self.y = 0
        self.indexBase = []

        ##############################
        ####### Image Database #######
        ##############################
        # Add each image (for evaluation) into a list (from sample code)
        imgFolder = imgFolder+'/*' + self.imgFormat
        print("[DEBUG] Folder to be indexed: ", imgFolder)
        for infile in glob(imgFolder):
            im = Image.open(infile)
            # Resize the image for thumbnails.
            imSize = im.size
            x = imSize[0]
            y = imSize[1]
            if x > self.x:
                self.x = x
            if y > self.y:
                self.y = y
            x = imSize[0] // 3
            y = imSize[1] // 3
            resized = im.resize((84, 96), Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(resized)

            # Find the max height and width of the set of pics.
            if x > self.xmax:
                self.xmax = x
            if y > self.ymax:
                self.ymax = y
            # Add the images to the lists.
            self.imageList.append(im)
            self.photoList.append(photo)
        if descriptor:
            # TODO: DONE  reading the saved descriptors
            # look for saved values
            #if os.path.isfile('indexBase.txt'):
            if withIndexBase:
                print("[INFO] descDist[0]+'_indexBase/*.csv' = ", descDist[0]+'_indexBase/*.csv')
                print("[INFO]-- Adding Images to the tree")
                data = []
                if descDist[0] == self.HISTOGRAMME_RGB: # Descriptor is Histogram
                    for index in glob(descDist[0]+'_indexBase/*.csv'):
                        data = csvmanager.readCSV_AVG(index)
                        # TODO: Add to M tree
                        csvHist = np.array(list(map(np.float32, data[1:])))
                        #print(csvHist)
                        hist = np.reshape(csvHist, (17, 17, 17))
                        self.addObjectsToTree([data[0], hist])
                        print(".", end= " ")
                elif descDist[0] == self.HISTOGRAMME_HSV: # Descriptor is Histogram
                    for index in glob(descDist[0]+'_indexBase/*.csv'):
                        data = csvmanager.readCSV_AVG(index)
                        # TODO: Add to M tree
                        csvHist = list(map(np.float32, data[1:]))
                        self.addObjectsToTree([data[0], csvHist])
                        print(".", end= " ")
                else:
                    for index in glob(descDist[0]+'_indexBase/*.csv'):
                        data = csvmanager.readCSV_AVG(index)
                        # TODO: Add to M tree
                        self.addObjectsToTree([data[0], data[1:]])
                        print(".", end= " ")
                print("\n[INFO]-- Insertion completed.")
                self.saveRawImagesFolder(data[0])
                

            # If not already computed, then compute the indexes
            else:
                # Compute
                print("[INFO]-- Adding Images to the tree")
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
                        print(".", end= " ")
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
                        print(".", end= " ")
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
                        print(".", end= " ")
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
                        print(".", end= " ")
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
                        print(".", end= " ")
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
                        print(".", end= " ")
                elif descDist[0] == self.COLOR_TEXTURE or descDist[0] == self.COLOR_SHAPE:
                    for im in self.imageList[:]:
                        # 1 get image data
                        fn, pixList = self.openImage(im)
                        fn = self.cleanFileName(im.filename)
                        image  = cv2.imread(im.filename.replace("\\","/"), cv2.IMREAD_GRAYSCALE)
                        grayImgData = cv2.resize(image, self.imgSize)
                        # 2 get descriptor
                        hu = [float(x) for x in descriptor(pixList, grayImgData)]
                        obj = [fn, hu]
                        # TODO: 3 Add to M tree
                        self.addObjectsToTree(obj)
                        # 4 Save to desk
                        self.saveToDesk([im.filename, hu])
                        print(".", end= " ")
            
                print("\n[INFO]-- Insertion completed.")

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
            csvmanager.writeCSV_AVG(folderName+fn+'.csv', data)
        else:
            mkdir(folderName)
            csvmanager.writeCSV_AVG(folderName+fn+'.csv', data)


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
        print("[INFO]-- Executing Image Search with k = ",k)
        query = [None, feature]
        result_list = list(self.mtree.k_NN_search(query, k))
        print("[INFO]-- Search finished")
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

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def getIndexBase(self):
        return self.indexBase
