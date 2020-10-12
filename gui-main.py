from gui import *
from imagemanager import *

if __name__ == '__main__':
    root = Tk()
    # root.resizable(widthFalse, height=False)
    root.iconbitmap("sidicbir.ico")
    root.title("Système de recherche d'image par le conrnu - CBIR")
    #root.attributes('-fullscreen', True)
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry("%dx%d" % (w, h))
    descDist = ["Moyenne Statistiques", "Euclidienne"]
    root.configure(bg='#e8e8e8')
    pix = ImageManager(root, None, None, descDist)
    top = CBIR_SIDI(root, pix, w, h)
    root.mainloop()


#pyinstaller.exe --onefile --icon=sidicbir.ico --paths guivenv\\Lib\\site-packages  -F --add-data "C:\\Windows\\System32\\vcomp140.dll;." --hidden-import=scipy.special.cython-special all.py

