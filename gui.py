from tkinter import *
from PIL import ImageTk, Image
from cbirtools import Descriptor
import cv2

class CBIR(Frame):
    def __init__(self, root, imageManager):
        # variables
        self.imgManager = imageManager # image manager
        self.imageList = imageManager.get_imageList()
        self.photoList = imageManager.get_photoList()
        self.indexBase = imageManager.getIndexBase()

        self.xmax = imageManager.get_xmax() + 20
        self.ymax = imageManager.get_ymax() + 10
        self.currentPhotoList = imageManager.get_photoList()
        self.currentImageList = imageManager.get_imageList()

        # COLORS
        self.bgc = '#ffffff' # background
        self.btc = '#161616' # buttons
        self.abtc = '#c6ffeb' # active button
        self.fgc = '#ffffff' # forground
        self.bth = 10 # button height
        self.currentPage = 0
        self.totalPages = self.get_totalPages()
        self.weights = []

        # main frame
        self.mainframe = Frame(root, bg=self.bgc)
        self.mainframe.pack()

        # section frames
        self.queryFrame = LabelFrame(self.mainframe, bg=self.bgc, text="Query section")
        self.queryFrame.pack(side=LEFT)

        self.resultsFrame = LabelFrame(self.mainframe, bg=self.bgc, text="Result section")
        self.resultsFrame.pack(side=RIGHT)

        # selected image window
        self.selectedImage = Label(self.queryFrame,
                                   width=450,
                                   height=imageManager.get_y(),
                                   bg=self.bgc)
        self.selectedImage.pack(side=TOP)
        self.update_preview(self.imageList[0].filename)

        # control panel (Buttons)
        self.controlPanel = Frame(self.queryFrame, bg=self.bgc)
        self.controlPanel.pack()

        self.btn_search = Button(self.controlPanel, text="Search",
                                 width=40, pady=self.bth, border=0, bg=self.btc,
                                 fg=self.fgc, activebackground=self.abtc,
                                 command=lambda: self.find())
        self.btn_search.pack(pady=10)

        self.btn_reset = Button(self.controlPanel, text="Reset",
                                width=40, pady=self.bth, border=0, bg=self.btc,
                                fg=self.fgc, activebackground=self.abtc,
                                command=lambda: self.reset())
        self.btn_reset.pack(pady=10)

        # _______________________________________________________________________ #
        # results frame
        self.resultsViewFrame = LabelFrame(self.resultsFrame, bg=self.bgc)
        self.resultsViewFrame.pack()

        instr = Label(self.resultsViewFrame,
                      bg=self.bgc, fg='#aaaaaa',
                      text="Click image to select. Checkboxes indicate relevance.")
        instr.pack()

        self.canvas = Canvas(self.resultsViewFrame,
                             bg="white",
                             highlightthickness=5)

        # page navigation
        self.pageButtons = Frame(self.resultsFrame, bg=self.bgc)
        self.pageButtons.pack()

        self.btn_prev = Button(self.pageButtons, text="<< Previous page",
                               width=30, border=0, bg=self.btc,
                               fg=self.fgc, activebackground=self.abtc,
                               command=lambda: self.prevPage())
        self.btn_prev.pack(side=LEFT)

        self.pageLabel = Label(self.pageButtons,
                               text="Page 1 of " + str(self.totalPages),
                               width=43, bg=self.bgc, fg='#aaaaaa')
        self.pageLabel.pack(side=LEFT)

        self.btn_next = Button(self.pageButtons, text="Next page >>",
                               width=30, border=0, bg=self.btc,
                               fg=self.fgc, activebackground=self.abtc,
                               command=lambda: self.nextPage())
        self.btn_next.pack(side=RIGHT)

        self.reset()

    def find(self):
        # TODO: use cv2
        #im = Image.open(self.selected.filename)
        #queryFeature = Descriptor.getHist(list(im.getdata()))

        im = cv2.imread(self.selected.filename)
        queryFeature = self.imgManager.descriptor(list(im))
        results = self.imgManager.executeImageSearch(queryFeature, 10)
        self.currentImageList, self.currentPhotoList = [], []
        for img in results:
            if img != 'None':
                im = Image.open(img.replace("'", ""))
                # Resize the image for thumbnails.
                #imSize = im.size
                #x = imSize[0] // 3
                #y = imSize[1] // 3
                #resized = im.resize((x, y), Image.ANTIALIAS)
                photo = ImageTk.PhotoImage(im)
                self.currentImageList.append(im)
                self.currentPhotoList.append(photo)

        iL = self.currentImageList[:20]
        pL = self.currentPhotoList[:20]
        self.currentPage = 0
        self.update_results((iL, pL))

   
    def reset(self):
        """
        Resets the GUI to its initial state
        """
        # initial display photos
        self.update_preview(self.imageList[0].filename)
        self.currentImageList = self.imageList
        self.currentPhotoList = self.photoList
        il = self.currentImageList[:20]
        pl = self.currentPhotoList[:20]
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
        cols = 5 # number of columns
        self.canvas.delete(ALL)
        self.canvas.config(width=(self.xmax) * 5, height=self.ymax * 4)
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
                imagesFrame.pack()

                # Put image as a button
                handler = lambda f=filename: self.update_preview(f)
                btn_image = Button(imagesFrame,
                                   image=img,
                                   border=0,
                                   bg="white",
                                   width=self.imgManager.get_xmax(),
                                   activebackground=self.bgc
                                   )
                btn_image.config(command=handler)
                btn_image.pack(side=LEFT)

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
        self.selected = Image.open(f.replace("\\", "/"))
        self.selectedPhoto = ImageTk.PhotoImage(self.selected)
        self.selectedImage.configure(image=self.selectedPhoto)

    # updates results page to previous page
    def prevPage(self):
        self.currentPage -= 1
        if self.currentPage < 0:
            self.currentPage = self.totalPages - 1
        start = self.currentPage * 20
        end = start + 20
        iL = self.currentImageList[start:end]
        pL = self.currentPhotoList[start:end]
        self.update_results((iL, pL))

    # updates results page to next page
    def nextPage(self):
        self.currentPage += 1
        if self.currentPage >= self.totalPages:
            self.currentPage = 0
        start = self.currentPage * 20
        end = start + 20
        iL = self.currentImageList[start:end]
        pL = self.currentPhotoList[start:end]
        self.update_results((iL, pL))

    # computes total pages in results
    def get_totalPages(self):
        pages = len(self.photoList) / 20
        if len(self.photoList) % 20 > 0:
            pages += 1
        return pages
