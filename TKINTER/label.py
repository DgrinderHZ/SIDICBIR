from tkinter import *

root = Tk()
root.title("Hello World!")
root.geometry("400x400")

my_lbl = Label(root, 
               text="I am a label!", 
               fg="white", 
               bg="black",
               font=("Helvetica", 22))
my_lbl.pack(pady=10)

lbl = Label(root, 
            text="Me too!", 
            relief="ridge",
            state="disabled",
            height=20,
            width=50)
lbl.pack()


root.mainloop()