from tkinter import *
from PIL import ImageTk, Image

root = Tk()
root.title("Hello World!")
root.geometry("400x400")
root.iconbitmap('c:/Users/elitebook/Documents/0.S4SIDI-PFE/gui-tkinter/widgets/icon.ico')

def clicked():
    text = e.get()
    my_label =Label(root, text="Hello "+text)
    my_label.pack()



# Crerate label
label = Label(root, text="Enter your name:")
label.pack(pady=5)

# Add images
my_image = ImageTk.PhotoImage(Image.open("logo.jpg"))
image_label = Label(image=my_image)
image_label.pack()

# Create inputField Entry
e = Entry(root, 
          width=100,
          font=("Helvetica", 18))
e.pack(pady=5)

# Create Buttons
my_button = Button(root, 
                   text="Click me!", 
                   command=clicked)
my_button.pack(pady=5)


root.mainloop()
