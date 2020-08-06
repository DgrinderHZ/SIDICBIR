from tkinter import *

root = Tk()
root.title("Hello World!")
root.geometry("400x400")

# Create labels
my_lbl = Label(root, 
               text="I am a label!", 
               fg="white", 
               bg="black",
               font=("Helvetica", 22))
my_lbl.grid(row=0, column=0, columnspan=2)

lbl = Label(root, 
            text="Me too!", 
            relief="ridge",
            state="disabled",
            height=2,
            width=7)
lbl.grid(row=1, column=0, sticky=E)

lbl2 = Label(root, 
            text="Me too! 2", 
            relief="ridge",
            state="disabled",
            height=2,
            width=7)
lbl2.grid(row=2, column=1, sticky=E)


root.mainloop()