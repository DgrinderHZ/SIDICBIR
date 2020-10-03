from tkinter import *
from glob import glob
from tkinter import filedialog
from PIL import ImageTk, Image
from cbirtools import *
from imagemanager import ImageManager
import cv2
from csvmanager import readCSV_AVG
from sklearn.metrics import confusion_matrix

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
        self.file_menu.add_command(label="Enregistrer résultat", command=None)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Effacer résultat", command=None)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Quitter", command=None)

        self.mymenu.add_command(label="Indexer", command=None)
        self.mymenu.add_command(label="Rechercher", command=None)

        self.view_menu = Menu(self.mymenu, tearoff=False)
        self.mymenu.add_cascade(label="Fenêtre", menu=self.view_menu)
        self.view_menu.add_command(label="Mesurer la qualité", command=lambda: self.mesurer())
        
        self.desc_menu = Menu(self.view_menu, tearoff=False)
        self.view_menu.add_cascade(label="Descripteurs", menu=self.desc_menu)
        self.desc_menu.add_command(label="Basé couleur", command=None)
        self.desc_menu.add_command(label="Basé texture", command=None)
        self.desc_menu.add_command(label="Basé forme", command=None)
        self.desc_menu.add_command(label="Composé", command=None)

        self.mymenu.add_command(label="Aide?", command=None)
        self.mymenu.add_command(label="À propos!", command=None)
        
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
                                     text="Query section")
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
        self.btn_browse = Button(self.canva_folder, text="Browse", command=self.browse_button)
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
    

        #__________________________________________________________________________________

        self.lbl_search = Label(self.dbQueryPanel, 
            text="OPTIONS: RECHERCHE M-TREE",
            fg="white", bg="gray",
            width=52,
            height=1,
            font=('Arial',10,'bold'))
        self.lbl_search.pack(pady=2)

        self.var_searchMethod = StringVar()
        self.var_searchMethod.set("Le nombre k : ")
        self.canva_SM = Canvas(self.dbQueryPanel, bg=self.bgc)
        self.canva_SM.pack()
        self.knn = Radiobutton(self.canva_SM, 
                                text="k-NN", 
                                bg=self.bgc,
                                variable=self.var_searchMethod,
                                value="Le nombre k : ",
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
                                    from_ = 0, to = 100, 
                                    textvariable=self.KRange)
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
                             bg=self.bgc,
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

        
    def mesurer(self):
        self.mesurUI = Toplevel(self.root)
        self.mesurUI.geometry("600x600")

        self.baseCanva = Canvas(self.mesurUI)
        self.baseCanva.pack(pady=5)
        self.label_base = Label(self.baseCanva,
                                bg=self.bgc,
                                text="Dossier de la base d'image utilsée:")
        self.label_base.grid(row=0, column=0)

        self.entryBase = Entry(self.baseCanva, width=50, textvariable=self.folder_path)
        self.entryBase.grid(row=0, column=1)

        self.classCanva = Canvas(self.mesurUI)
        self.classCanva.pack(pady=5)
        self.label_queryClass = Label(self.classCanva,
                                bg=self.bgc,
                                text="Classe d'image requête:")
        self.label_queryClass.grid(row=0, column=0)

        self.classe = StringVar()
        self.classe.set(self.imgManager.cleanFileName(self.selected.filename))
        self.entryQueryClass = Entry(self.classCanva, width=20, textvariable=self.classe)
        self.entryQueryClass.grid(row=0, column=1)
        self.btn_browseQC = Button(self.mesurUI, text="Confusion matrix", 
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
        print(cm)
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
                                 text="Calculer!",
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
            print("[INFO] SELECTED FOLDER: empty path")
        else:
            print("[INFO] SELECTED FOLDER: ", self.folder_path.get())
        
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
        elif self.imgManager.descDist[0] == self.COLOR_TEXTURE or \
            self.imgManager.descDist[0] == self.COLOR_SHAPE:
            # 1 get image data
            fn, pixList = self.imgManager.openImage(self.selected)
            path = self.selected.filename.replace("\\","/")
            # 2 get descriptor
            queryFeature  = [float(x) for x in self.imgManager.descriptor(pixList, path)]
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
            queryFeature = self.imgManager.descriptor(im)
        print(len(queryFeature))
        self.results = self.imgManager.executeImageSearch(queryFeature, self.KRange.get())
        self.currentImageList, self.currentPhotoList = [], []
        self.resultsLenght = 0
        for img in self.results:
            if img != 'None':
                self.resultsLenght += 1
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
    
    
