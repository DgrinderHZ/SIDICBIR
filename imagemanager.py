
import glob, os
from PIL import ImageTk, Image
from mtree import *
import cv2
import csvmanager

# Pixel Info class (from sample code)
class ImageManager:
    def __init__(self, root, descriptor, distance, descDist, imgFolder, imageFormat, withIndexBase=False):
        self.root = root
        self.descriptor = descriptor
        self.distance = distance
        self.imgFolder = imgFolder
        self.descDist = descDist
        ##############################
        ########### Mtree ############
        ##############################
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
        if withIndexBase:
            imgFolder = 'default/*.jpg'
        else:
            imgFolder = imgFolder+"/*.jpg"
        for infile in glob.glob(imgFolder):
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

        # TODO: DONE  reading the saved descriptors
        # look for saved values
        #if os.path.isfile('indexBase.txt'):
        if withIndexBase:
            print("[INFO]-- Adding Images to the tree")
            for index in glob.glob('indexBase/*.csv'):
                data = csvmanager.readCSV_AVG(index)
                # TODO: Add to M tree
                self.addObjectsToTree([data[0], data[1:]])
            print("\n[INFO]-- Insertion completed.")
            #signature = open(index, 'r')
            '''for line in signature:
                line = line.replace("[", "").replace("]", "").replace("(", "").replace(")", "").split(",")
                filename = line[0]
                avgs = [float(x) for x in line[1:]]
                # TODO: Add to M tree
                self.addObjectsToTree([filename, avgs])
                print(".", end= " ")'''

        # If not already computed, then compute the indexes
        else:
            # Compute
            print("[INFO]-- Adding Images to the tree")
            if descDist[0] == "Avgs":
                for im in self.imageList[:]:
                    # 1 get image data
                    fn, pixList = self.openImage(im)
                    # 2 get descriptor
                    avgs = [float(x) for x in descriptor(pixList)]
                    obj = [fn, avgs]
                    # 3 Save to desk
                    self.saveToDesk(obj)
                    # TODO: 4 Add to M tree
                    self.addObjectsToTree(obj)
                    print(".", end= " ")
            elif descDist[0] == "Hist": # Descriptor is Histogram
                for im in self.imageList[:]:
                    # 1 get image data
                    fn, pixList = self.openImage(im)
                    # 2 get descriptor
                    hist = descriptor(pixList)
                    obj = [fn, hist]
                    # 3 Save to desk
                    self.saveToDesk(obj)
                    # TODO: 4 Add to M tree
                    self.addObjectsToTree(obj)
                    print(".", end= " ")
            print("\n[INFO]-- Insertion completed.")

            # Save to desk
            '''
            indexes = open('indexBase.txt', 'w')
            for i in range(len(self.indexBase)):
                indexes.write(str(self.indexBase[i]))
                indexes.write("\n")
            indexes.close()
            '''
    def openImage(self, im):
        fn = self.cleanFileName(im.filename)
        imData = cv2.imread(im.filename.replace("\\","/"))
        imData = cv2.resize(imData, (24, 24))
        pixList = list(imData)
        return fn, pixList

    def saveToDesk(self, obj):
        fn = self.cleanFileName(obj[0])
        data = [fn] # filename
        data.extend(obj[1]) # avgs, hist, moment, ...
        csvmanager.writeCSV_AVG('indexBase/'+fn+'.csv', data)

    def cleanFileName(self, filename):
        fn = filename
        p = fn[::-1].find("\\")
        if p != -1:
            fn = fn[len(fn)-p:]
        else:
            p = fn[::-1].find("/")
            if p != -1:
                fn = fn[len(fn)-p:]
        return fn

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
