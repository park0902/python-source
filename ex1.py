# from tkinter import *
#
# tk = Tk()
# canvas = Canvas(tk, width=500, height=450)
# canvas.pack()
#
#
# canvas.create_polygon(0,350,15,350,15,375,30,375,30,350,45,350,45,450,0,450, outline='yellow', fill='green', width=1)
# canvas.mainloop()



from tkinter import *
import random
import time



class Man:
    def __init__(self, canvas):
        self.canvas = canvas
        self.man = canvas.create_rectangle(0, 0, 10, 20, fill='black')
        self.canvas.move(self.man, 295, 480)
        self.x = 0
        self.y = 0
        self.canvas_width = self.canvas.winfo_width()      #man의 이동범위를 canvas의 밑변으로 제한
        self.canvas_height = self.canvas.winfo_height()    #man의 이동범위를 canvas의 높이로 제한
        self.canvas.bind_all('<KeyPress-Left>', self.turn_left)
        self.canvas.bind_all('<KeyPress-Right>', self.turn_right)   #canvas가 감지
        print(self.canvas_width)

    def draw(self):
        man_pos = self.canvas.coords(self.man)       #self.man의 좌상우하의 좌표, 위치

        if man_pos[0] <= 0 and self.x < 0:                      #self.man이 오른쪽으로 나가지 않도록
            self.x = 5
        elif man_pos[2] >= self.canvas_width and self.x > 0:
            self.x = -5

        self.canvas.move(self.man, self.x, self.y)


    def turn_left(self, evt):
        self.x = -5


    def turn_right(self, evt):
        self.x = 5

#
# class poo:
#     def __init__(self):
#
# class candy:
#     def __init__(self):


tk = Tk()
tk.title("Dodge Your Poop Faster")   #게임 창의 제목 출력
tk.resizable(0, 0)                   #tk.resizable(가로크기조절, 세로크기조절)
tk.wm_attributes("-topmost", 1)      #생성된 게임창을 다른창의 제일 위에 오도록 정렬
tk.update()  # 여기서 한번 다시 적어준다.

canvas = Canvas(tk, width=600, height=500, bd=0, highlightthickness=0)
#bd=0, highlightthickness=0 은 베젤의 크기를 의미한다.
canvas.configure(background='#E8D487')
canvas.pack()  #앞의 코드에서 전달된 폭과 높이는 매개변수에 따라 크기를 맞추라고 캔버스에에 말해준다.

man = Man(canvas)

while 1:
    tk.update()
    tk.update_idletasks()
    man.draw()
    time.sleep(0.015)