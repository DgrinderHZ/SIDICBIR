from tkinter import *
from glob import glob
from tkinter import filedialog
from PIL import ImageTk, Image
from cbirtools import *
from imagemanager import ImageManager
import cv2
from csvmanager import readCSV_AVG

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

        self.xmax = imageManager.get_xmax() + 20
        self.ymax = imageManager.get_ymax() + 10
        self.currentPhotoList = imageManager.get_photoList()
        self.currentImageList = imageManager.get_imageList()

        # COLORS
        self.bgc = '#ffffff' # background
        self.btc = '#15CCF4' # buttons
        self.abtc = '#c6ffeb' # active button
        self.fgc = '#ffffff' # forground
        self.basedOnC = ["#CF1F1F", "grey", "grey", "grey"]
        self.bth = 5 # button height
        self.currentPage = 0
        self.totalPages = self.get_totalPages()
        self.weights = []

        # main frame
        #self.mainframe = Frame(root, bg=self.bgc, width=1366,  height=768)
        #self.mainframe.pack()

        self.mymenu = Menu(self.root)
        self.root.config(menu=self.mymenu)  
        self.file_menu = Menu(self.mymenu)
        self.mymenu.add_cascade(label="File", menu=self.file_menu)
        self.new_menu = Menu(self.file_menu)
        self.file_menu.add_cascade(label="New", menu=self.new_menu)
        self.new_menu.add_command(label="Image database", command=self.browse_button)
        self.new_menu.add_command(label="Query image", command=self.browse_buttonQ)

        
        self.upperFrame = Frame(self.root)
        self.upperFrame.pack()

        self.btn_color = Button(self.upperFrame, 
                                 text="Coulor Based",
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
                                 text="Texture Based",
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
                                 text="Shape Based",
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
                                 text="Fusion (Colur+Texture/Shape)",
                                 font=('Arial',10,'bold'),
                                 width=41, 
                                 pady=self.bth, 
                                 border=5, 
                                 bg=self.basedOnC[3],
                                 fg=self.fgc,  
                                 activebackground=self.abtc,
                                 command=lambda: self.contentType(4))
        self.btn_fusion.grid(row=0, column=3)

        self.lowerFrame = Frame(self.root, bg="white")
        self.lowerFrame.pack()
        
        #############################
        ####### Query Frame #########
        #############################


        self.queryFrame = LabelFrame(self.lowerFrame, 
                                     bg=self.bgc, 
                                     height=768,
                                     text="Query section")
        self.queryFrame.grid(row=0, column=0)

        # control panel (Buttons)
        self.dbQueryPanel = Frame(self.queryFrame, bg=self.bgc)
        self.dbQueryPanel.pack()

        # Database control
        self.lbl_dbQueryPanel = Label(self.dbQueryPanel, 
            text="OPTIONS: INDEXATION ET BASES DE DONNEES (OFFLINE)",
            fg="black",
            width=52,
            height=1,
            font=('Arial',10,'bold'))
        self.lbl_dbQueryPanel.pack()

        self.var_choix = StringVar()
        self.canva_BDD = Canvas(self.dbQueryPanel, bg="white")
        self.canva_BDD.pack()
        self.rdio_ByImages = Radiobutton(self.canva_BDD, 
                                        text="Dossier d'images", 
                                        bg="white",
                                        variable=self.var_choix,
                                        value="Dossier d'images : ", 
                                        width=25)

        self.rdio_ByCSVs = Radiobutton(self.canva_BDD, 
                                        text="Dossier CSVs", 
                                        bg="white",
                                        variable=self.var_choix,
                                        value="Dossier CSVs : ",
                                        width=25)
        self.var_choix.set("Dossier d'images : ")
        self.rdio_ByImages.grid(row=0, column=0)
        self.rdio_ByCSVs.grid(row=0, column=1)
        
        ######## Folder selection
        self.folder_path = StringVar()
        self.canva_folder = Canvas(self.dbQueryPanel, bg="white")
        self.canva_folder.pack()

        self.label_folder = Label(self.canva_folder, 
                                  textvariable=self.var_choix,
                                  bg="white")
        self.label_folder.grid(row=0, column=0)
        self.entryFolder = Entry(self.canva_folder, width=40, textvariable=self.folder_path)
        self.entryFolder.grid(row=0, column=1)
        self.btn_browse = Button(self.canva_folder, text="Browse", command=self.browse_button)
        self.btn_browse.grid(row=0, column=2)
        
        #________________________________________________________
        
        self.offlineOptions = Canvas(self.dbQueryPanel, bg="white")
        self.offlineOptions.pack()
        ######## Descriptor selection
        self.var_desciptor = StringVar()
        optionList_desc = ('Moments_Staistiques', 'Histogramme', 'Avgs')
        self.canva_desc = Canvas(self.offlineOptions, bg="white")
        self.canva_desc.grid(row=0, column=0)

        self.label_desc = Label(self.canva_desc, 
                                bg="white",
                                text="Descripteur : ")
        self.label_desc.grid(row=0, column=0)
        self.var_desciptor.set(optionList_desc[0])
        self.om_descriptor = OptionMenu(self.canva_desc, 
                            self.var_desciptor,
                             *optionList_desc
                    )
        self.om_descriptor.grid(row=0, column=1)
        #___________________________________________________________________________#
        
        ######## Type of image selection
        self.var_typeImg = StringVar()
        optionList_ti = ('.jpg', '.png')
        self.canva_typeImg = Canvas(self.offlineOptions, bg="white")
        self.canva_typeImg.grid(row=0, column=1)

        self.label_typeImg = Label(self.canva_typeImg, 
                                    text="Type d'images : ",
                                    bg="white")
        self.label_typeImg.grid(row=0, column=0)
        self.var_typeImg.set(optionList_ti[0])
        self.om_imgType = OptionMenu(self.canva_typeImg, self.var_typeImg, *optionList_ti)
        self.om_imgType.grid(row=0, column=1)
        
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
            fg="black",
            width=52, height=1,
            font=('Arial',10,'bold'))
        self.lbl_descriptrs.pack()
        self.query_path = StringVar()
        self.canva_query = Canvas(self.dbQueryPanel, bg="white")
        self.canva_query.pack()

        self.label_query = Label(self.canva_query,
                                bg="white",
                                text="Image requête:")
        self.label_query.grid(row=0, column=0)
        self.entryQuery = Entry(self.canva_query, width=40, textvariable=self.query_path)
        self.entryQuery.grid(row=0, column=1)
        self.btn_browseQ = Button(self.canva_query, text="Browse", command=self.browse_buttonQ)
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
                                 text="Search",
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
                                text="Reset",
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
    

        self.lbl_distances = Label(self.dbQueryPanel, 
            text="OPTIONS: CHOIX DE LA MESURE DE SIMILARITE",
            fg="black",
            width=52, height=1,
            font=('Arial',10,'bold'))
        self.lbl_distances.pack()

        ######## Distance selection
        self.var_distance = StringVar()
        optionListD = ('D. Euclidien', 'CHi square', 'Interesect', 'Manhatan', 'ManhatanFusion')
        self.canva_dist = Canvas(self.dbQueryPanel, bg="white")
        self.canva_dist.pack()

        self.label_dist = Label(self.canva_dist, 
                                bg="white",
                                text="Choix du distance : ")
        self.label_dist.grid(row=0, column=0)
        self.var_distance.set(optionListD[0])
        self.omd = OptionMenu(self.canva_dist, self.var_distance, *optionListD)
        self.omd.grid(row=0, column=1)

        #__________________________________________________________________________________

        self.lbl_search = Label(self.dbQueryPanel, 
            text="OPTIONS: RECHERCHE M-TREE",
            fg="black",
            width=52,
            height=1,
            font=('Arial',10,'bold'))
        self.lbl_search.pack()

        self.var_searchMethod = StringVar()
        self.var_searchMethod.set("Le nombre k : ")
        self.canva_SM = Canvas(self.dbQueryPanel, bg="white")
        self.canva_SM.pack()
        self.knn = Radiobutton(self.canva_SM, 
                                text="k-NN", 
                                bg="white",
                                variable=self.var_searchMethod,
                                value="Le nombre k : ",
                                width =25)
        self.rquery = Radiobutton(self.canva_SM, 
                                  text="Range query", 
                                  bg="white",
                                  variable=self.var_searchMethod,
                                  value="Le rayon r : ",
                                  width =25)
        self.knn.grid(row=0, column=0)
        self.rquery.grid(row=0, column=1)

        self.KRange = IntVar()
        self.KRange.set(10)
        self.canva_KRange = Canvas(self.dbQueryPanel, bg="white")
        self.canva_KRange.pack()
        self.label_KRange = Label(self.canva_KRange, 
                                    bg="white",
                                    textvariable=self.var_searchMethod)
        self.label_KRange.grid(row=0, column=0)
        self.entryKRange = Spinbox(self.canva_KRange, width=4, from_ = 0, to = 100, textvariable=self.KRange)
        self.entryKRange.grid(row=0, column=1)


        # _______________________________________________________________________ #
        ##################################
        # results frame
        ##################################
        
        self.resultsViewFrame = LabelFrame(self.lowerFrame, 
                                           bg=self.bgc, 
                                           text="Result section")
        self.resultsViewFrame.grid(row=0, column=1)

        instr = Label(self.resultsViewFrame,
                      bg=self.bgc, 
                      fg='#aaaaaa',
                      text="Click image to select. Checkboxes indicate relevance.")
        instr.pack()

        self.resultPanel = LabelFrame(self.resultsViewFrame, bg=self.bgc)
        self.resultPanel.pack()
        self.canvas = Canvas(self.resultPanel ,
                             bg="white",
                             width=920,
                             height=520
                            )
        self.canvas.pack()

        # page navigation
        self.pageButtons = Frame(self.resultsViewFrame, bg=self.bgc)
        self.pageButtons.pack()

        self.btn_prev = Button(self.pageButtons, text="<< Previous page",
                               width=30, border=5, bg=self.btc,
                               fg=self.fgc, activebackground=self.abtc,
                               command=lambda: self.prevPage())
        self.btn_prev.pack(side=LEFT)

        self.pageLabel = Label(self.pageButtons,
                               text="Page 1 of " + str(self.totalPages),
                               width=43, bg=self.bgc, fg='#aaaaaa')
        self.pageLabel.pack(side=LEFT)

        self.btn_next = Button(self.pageButtons, text="Next page >>",
                               width=30, border=5, bg=self.btc,
                               fg=self.fgc, activebackground=self.abtc,
                               command=lambda: self.nextPage())
        self.btn_next.pack(side=RIGHT)

        

    def contentType(self, code):
        self.basedOnC[code-1], self.basedOnC[self.basedOn-1] = self.basedOnC[self.basedOn-1], self.basedOnC[code-1]
        self.btn_color.config(bg=self.basedOnC[0])
        self.btn_texture.config(bg=self.basedOnC[1])
        self.btn_shape.config(bg=self.basedOnC[2])
        self.btn_fusion.config(bg=self.basedOnC[3])

        self.basedOn = code
        self.reset()

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
        if self.folder_path.get() == "":
            print("[INFO] SELECTED FOLDER: empty path")
        else:
            print("[INFO] SELECTED FOLDER: ", self.folder_path.get())
        
        # **************** CHOIX: DESCRIPTOR n DISTANCE *****************
        colorDescriptor = ColorDescriptor()
        textureDescriptor = TextureDescriptor()
        shapeDescriptor = ShapeDescriptor()
        fusionDescriptors = FusionDescriptors(0.5, 0.5)
        distance = Distance()

        DESC = colorDescriptor.getAvgs
        DIST = distance.euclid
        descDist = ["Avgs", "Euclid"]

        if self.var_desciptor.get() == 'Moments_Staistiques':
            DESC = colorDescriptor.getMoments
            descDist[0] = "Moments_Staistiques"
            print("[INFO] DESC = Moments_Staistiques")
        elif self.var_desciptor.get() == 'Histogramme':
            DESC = colorDescriptor.getHist
            descDist[0] = "Hist"
            print("[INFO] DESC = Hist")
        elif self.var_desciptor.get() == 'Avgs':
            DESC = colorDescriptor.getAvgs
            descDist[0] = "Avgs"
            print("[INFO] DESC = Avgs")
        elif self.var_desciptor.get() == 'Gabor':
            DESC = textureDescriptor.getGabor
            descDist[0] = "Gabor"
            print("[INFO] DESC = Gabor")
        elif self.var_desciptor.get() == 'GaborV':
            DESC = textureDescriptor.getGaborFeatures
            descDist[0] = "GaborV"
            print("[INFO] DESC = GaborV")
        elif self.var_desciptor.get() == 'Haralick':
            DESC = textureDescriptor.getHaralickFeatures
            descDist[0] = "Haralick"
            print("[INFO] DESC = Haralick")
        elif self.var_desciptor.get() == 'HuMoments':
            DESC = shapeDescriptor.getHuMoments
            descDist[0] = "HuMoments"
            print("[INFO] DESC = HuMoments")
        elif self.var_desciptor.get() == 'ZernikeMoments':
            DESC = shapeDescriptor.getZernikeMoments
            descDist[0] = "ZernikeMoments"
            print("[INFO] DESC = ZernikeMoments")
        elif self.var_desciptor.get() == "MomentsStat + Gabor":
            DESC = fusionDescriptors.getMomentsAndGabor
            descDist[0] = "MomentsStat + Gabor"
            print("[INFO] DESC = MomentsAndGabor")
        elif self.var_desciptor.get() == "MomentsStat + Zernike":
            DESC = fusionDescriptors.getMomentsAndZernike
            descDist[0] = "MomentsStat + Zernike"
            print("[INFO] DESC = MomentsAndZernike")

        
        if self.var_distance.get() == 'D. Euclidien':
            DIST = distance.euclid
            descDist[1] = "Euclid"
            print("[INFO] DIST = Euclid")
        elif self.var_distance.get() == 'CHi square':
            DIST = distance.chi
            descDist[1] = "Chi2"
            print("[INFO] DIST = CHISQ2")
        elif self.var_distance.get() == 'Interesect':
            DIST = distance.intersect
            descDist[1] = "Intersect"
            print("[INFO] DIST = Intersect")
        elif self.var_distance.get() == 'Interesect':
            DIST = distance.intersect
            descDist[1] = "Intersect"
            print("[INFO] DIST = Intersect")
        elif self.var_distance.get() == 'Manhatan':
            DIST = distance.manhatan
            descDist[1] = "Manhatan"
            print("[INFO] DIST = Manhatan")
        elif self.var_distance.get() == 'ManhatanFusion':
            DIST = fusionDescriptors.manhatanDistance
            descDist[1] = "ManhatanFusion"
            print("[INFO] DIST = ManhatanFusion")
        
       
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

        self.imgManager = ImageManager(self.root, DESC, DIST, descDist, imgFolder, imageFormat, self.withIndexBase)
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
        if self.imgManager.descDist[0] == 'ZernikeMoments':
            # 1 get image data
            image  = cv2.imread(self.selected.filename.replace("\\","/"), cv2.IMREAD_GRAYSCALE)
            imData = cv2.resize(image, (32, 32))
            # 2 get descriptor
            queryFeature = [float(x) for x in self.imgManager.descriptor(imData)]
        elif self.imgManager.descDist[0] == "MomentsStat + Gabor" or self.imgManager.descDist[0] == "MomentsStat + Zernike":
            # 1 get image data
            path = self.selected.filename.replace("\\","/")
            fn, pixList = self.imgManager.openImage(self.selected)
            image  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            grayImgData = cv2.resize(image, (32, 32))
            # 2 get descriptor
            queryFeature  = [float(x) for x in self.imgManager.descriptor(pixList, grayImgData)]
        elif 'Gabor' in self.imgManager.descDist[0] or self.imgManager.descDist[0] == 'HuMoments':
            # 1 get image data
            image  = cv2.imread(self.selected.filename.replace("\\","/"), cv2.IMREAD_GRAYSCALE)
            imData = cv2.resize(image, (32, 32))
            # 2 get descriptor
            queryFeature = [float(x) for x in self.imgManager.descriptor(imData)]
        else:
            im = cv2.imread(self.selected.filename)
            im = cv2.resize(im, (32, 32))
            queryFeature = self.imgManager.descriptor(list(im))
        
        results = self.imgManager.executeImageSearch(queryFeature, self.KRange.get())
        self.currentImageList, self.currentPhotoList = [], []
        for img in results:
            if img != 'None':
                if self.withIndexBase:
                    im = Image.open(img) #.replace("/", "")
                else:
                    im = Image.open(self.imgManager.imgFolder + "/" + img) #.replace("/", "")
                # Resize the image for thumbnails.
                resized = im.resize((128, 128), Image.ANTIALIAS)
                photo = ImageTk.PhotoImage(resized)
                self.currentImageList.append(im)
                self.currentPhotoList.append(photo)

        iL = self.currentImageList[:24]
        pL = self.currentPhotoList[:24]
        self.currentPage = 0
        self.update_results((iL, pL))

   
    def reset(self):
        """
        Resets the GUI to its initial state
        """
        # initial display photos
        if self.basedOn == 1:
            optionList_desc = ('Moments_Staistiques', 'Histogramme', 'Avgs')
        elif self.basedOn == 2:
            optionList_desc = ('Gabor', 'GaborV', 'Haralick')
        elif self.basedOn == 3:
            optionList_desc = ('HuMoments', 'ZernikeMoments')
        elif self.basedOn == 4:
            optionList_desc = ('MomentsStat + Gabor', 'MomentsStat + Zernike')
        
        self.var_desciptor.set(optionList_desc[0])
        self.om_descriptor.destroy()

        self.om_descriptor = OptionMenu(self.canva_desc, 
                                        self.var_desciptor, 
                                        *optionList_desc)
        self.om_descriptor.grid(row=0, column=1)
        
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
                                   bg="white",
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
        self.selected = Image.open(f.replace("\\", "/"))
        resized = self.selected.resize((320, 200), Image.ANTIALIAS)
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
    
    
