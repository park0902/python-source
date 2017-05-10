from tkinter import *
import random
import time

class Ball:

    def __init__(self, canvas, paddle, color):
        self.canvas = canvas
        self.paddle = paddle
        self.id = canvas.create_oval(10, 10, 25, 25, fill=color) #공 좌표 및 색깔(oval : object 형태 타입)
        self.canvas.move(self.id, 245, 100) #공을 캔버스 중앙으로 이동
        starts = [-3, -2, -1, 1, 2, 3]
        random.shuffle(starts)
        #공의 속도
        self.x = starts[0]
        self.y = -3
        self.canvas_height = self.canvas.winfo_height()
        self.canvas_width = self.canvas.winfo_width()
        self.hit_bottom = False

    def hit_paddle(self,pos):
        paddle_pos = self.canvas.coords(self.paddle.id)
        if pos[2] >= paddle_pos[0] and pos[0] <= paddle_pos[2]:
            if pos[3] >= paddle_pos[1] and pos[3] <= paddle_pos[3]:
                return True
        return False

    def draw(self):
        self.canvas.move(self.id, self.x, self.y) #공을 움직이게 하는 부분
        #공이 화면 밖으로 나가지 않게 해준다
        pos = self.canvas.coords(self.id)
        if pos[1] <= 0:
            self.y = 3
        if pos[3] >= self.canvas_height: #바닥에 부딪히면 게임오버
            self.hit_bottom = True
        if pos[0] <= 0:
            self.x = 3
        if pos[2] >= self.canvas_width:
            self.x = -3
        if self.hit_paddle(pos) == True: #판에 부딪히면 위로 튕겨올라가게
            self.y = -3

class Paddle:

    def __init__(self,canvas,color):
        self.canvas = canvas
        self.id = canvas.create_rectangle(0,0,40,10,fill=color)
        self.canvas.move(self.id, 200, 300)
        self.x = 0
        self.canvas_width = self.canvas.winfo_width()
        self.canvas.bind_all('<KeyPress-Left>',self.turn_left)
        self.canvas.bind_all('<KeyPress-Right>',self.turn_right)

    def draw(self):
        self.canvas.move(self.id, self.x, 0)
        pos = self.canvas.coords(self.id)
        if pos[0] <= 0:
            self.x = 0
        elif pos[2] >= self.canvas_width:
            self.x = 0

    def turn_left(self,evt):
        self.x = -9

    def turn_right(self,evt):
        self.x = 9

tk = Tk()
tk.title("Game")
tk.resizable(0, 0)
tk.wm_attributes("-topmost", 1)
canvas = Canvas(tk, width=600, height=500, bd=0, highlightthickness=0)
canvas.pack()
tk.update()
paddle = Paddle(canvas,'blue')
ball = Ball(canvas, paddle, 'red')
start = False

#공을 약간 움직이고 새로운 위치로 화면을 다시 그리며, 잠깐 잠들었다가 다시 시작해라!
while 1:
    if ball.hit_bottom == False:
        ball.draw()
        paddle.draw()
    #그림을 다시 그려라! 라고 쉴새없이 명령
    tk.update_idletasks()
    tk.update()
    #시작 전 2초간 sleep
    if not start:
        time.sleep(2)
        start = True
    time.sleep(0.03)