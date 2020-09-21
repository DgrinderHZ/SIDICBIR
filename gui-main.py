from gui import *
from imagemanager import * 
from cbirtools import *

if __name__ == '__main__':
    root = Tk()
    # root.resizable(width=False, height=False)
    root.iconbitmap("fsts.ico")
    root.title("Système de recherche d'image par le conrnu - CBIR")
    #root.attributes('-fullscreen', True)
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    print(w, h )
    root.geometry("%dx%d+0+0" % (w, h))
    descDist = ["Avgs", "Euclid"]
    
    root.configure(bg='white')
    pix = ImageManager(root, Descriptor.getAvgs, Distance.euclid, descDist)
    top = CBIR_SIDI(root, pix, w, h)
    root.mainloop()

