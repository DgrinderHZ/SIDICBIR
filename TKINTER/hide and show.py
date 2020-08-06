from tkinter import *

root = Tk()
root.title("Hello World!")
root.geometry("400x400")

def clicked():
    global my_label
    text = e.get()
    my_label =Label(root, text="Hello "+text)
    my_label.pack()

def hide():
    my_label.pack_forget()
    #my_label.destroy()

def show():
    my_label.pack()
# Crerate label
label = Label(root, text="Enter your name:")
label.pack(pady=5)


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

my_button = Button(root, 
                   text="Hide", 
                   command=hide)
my_button.pack(pady=5)

my_button = Button(root, 
                   text="Show", 
                   command=show)
my_button.pack(pady=5)


root.mainloop()
