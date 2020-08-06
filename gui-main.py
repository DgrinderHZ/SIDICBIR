from gui import *
from imagemanager import *
from cbirtools import *

if __name__ == '__main__':
    root = Tk()
    # root.resizable(width=False, height=False)
    pix = ImageManager(root, Descriptor.getAvgs, Distance.euclid, "images/*.jpg")
    top = CBIR(root, pix)
    root.mainloop()

