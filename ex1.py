from tkinter import *

tk = Tk()
canvas = Canvas(tk, width=500, height=450)
canvas.pack()


canvas.create_polygon(0,350,15,350,15,375,30,375,30,350,45,350,45,450,0,450, outline='yellow', fill='green', width=1)
canvas.mainloop()