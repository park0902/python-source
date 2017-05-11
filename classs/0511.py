from tkinter import *
import random
import time

class Ball:

    def __init__(self, canvas, paddle, color):
        self.canvas = canvas
        self.paddle = paddle
        self.id = canvas.create_oval(10, 10, 25, 25, fill=color) #공 좌표 및 색깔(oval : object 형태 타입)
        self.canvas.move(self.id, 250, 0)   #공을 캔버스 중앙으로 이동
        # 캔버스 공 위치 선정 공 , x축 , y축
        canvas.configure(background='black')    # 캔버스 바탕을 black 으로
        starts = [-4,-3,-2,-1,1,2,3,5,6,7]   # 음수만 있을때는 왼쪽, 양수만 있을때는 오른쪽, 0은 중간으로
        random.shuffle(starts)
        #공의 속도
        self.x = 0  # starts[0]
        self.y = 0   # 음수일때 공이 위로 올라가고 양수일때 공이 밑으로 내려간다.
        self.canvas_height = self.canvas.winfo_height()   # 캔버스의 높이
        self.canvas_width = self.canvas.winfo_width()     # 캔버스의 넓이
        self.hit_bottom = False                         # 공이 바닥에 닿으면 게임이 끝나는 코드를 구현하기 위해 쓴다
        # self.canvas.bind_all('<KeyPress-Left>', self.turn_left)  # keyPress로 패들 좌우 지정
        # self.canvas.bind_all('<KeyPress-Right>', self.turn_right)
        # self.canvas.bind_all('<KeyPress-Up>', self.turn_up)  # keyPress로 패들 좌우 지정
        # self.canvas.bind_all('<KeyPress-Down>', self.turn_down)
        self.canvas.bind_all('<space>', self.turn_up)
        # self.canvas.bind_all('<Key>', self.turn_up)
    # def turn_left(self, evt):
    #         self.x = -9
    #
    # def turn_right(self, evt):
    #         self.x = 9

    def turn_up(self, evt):
        starts = [-6,-5,-4, -3, -2, -1, 1, 2, 3, 5, 6, 7]
        random.shuffle(starts)
        self.y = -9
        self.x = starts[0]
        self.winner = True
    #
    # def turn_down(self, evt):
    #     self.y = 9
    def hit_paddle(self,pos):

        paddle_pos = self.canvas.coords(self.paddle.id)
        # print(paddle_pos)
        # if pos[3] == paddle_pos[3] and self.y > 0:
        #     return -1

        if pos[2] >= paddle_pos[0] and pos[0] <= paddle_pos[2]:  #공의 동쪽이 패들의 서쪽보다 크고 공의 서쪽이 패들의동쪽보다 작을때
            if pos[3] >= paddle_pos[1] and pos[1] <= paddle_pos[3]:  # 공의 북쪽이 패들의 남쪽보다 크고 공의 남쪽이 패들의 북쪽보다 작을때
                return True
        return False

    def gameover(self):
        pos = self.canvas.coords(self.id)
        paddle_pos = self.canvas.coords(self.paddle.id)
        if pos[3] == paddle_pos[3] and self.y > 0:
            return -1
        elif self.hit_paddle(pos) ==  True:
            return 1
        else:
            return 0
        return

    def winner(self,winner):
        return winner


    def draw(self):
        self.canvas.move(self.id, self.x, self.y)                     #공을 움직이게 하는 부분
        #공이 화면 밖으로 나가지 않게 해준다 # 공, 공의 x방향(좌우), 공의 y방향(위아래)
        pos = self.canvas.coords(self.id)   # 공의 위치
        # print(pos) #서,남,동,북 순의 리스트
        if pos[1] <= 0:         # 남쪽 좌표
            self.y = 3
        if pos[3] >= self.canvas_height:    #북쪽 좌표
            # self.hit_bottom = True          #바닥에 부딪히면 게임오버
            self.y = -3
        if pos[0] <= 0:                      #서쪽 좌표
            self.x = 3
        if pos[2] >= self.canvas_width:        # 동쪽 좌표
            self.x = -3
        if self.hit_paddle(pos) ==  True:                          #판에 부딪히면 위로 튕겨올라가게
            self.y = -3





class Paddle:

    def __init__(self,canvas,color):
        self.canvas = canvas
        self.id = canvas.create_rectangle(0,0,80,10,fill=color)       # 패들 크기
        self.canvas.move(self.id, 300, 300)  # 300은 width 위치 , 450은 height 위치
        self.x = 0             # 게임 시작시 움직이 말라고 속도를 고정시킴
        self.canvas_width = self.canvas.winfo_width() # 화면밖으로 패들이 나가지 않도록
        self.canvas.bind_all('<KeyPress-Left>',self.turn_left)         # keyPress로 패들 좌우 지정
        self.canvas.bind_all('<KeyPress-Right>',self.turn_right)
        # self.canvas.bind_all('<KeyPress-Up>', self.turn_up)  # keyPress로 패들 좌우 지정
        # self.canvas.bind_all('<KeyPress-Down>', self.turn_down)
        # self.y=  0
    def draw(self):

        pos = self.canvas.coords(self.id)
        if pos[0] <= 0 and self.x < 0:     # 패들의 서쪽이 0보다 작다면
            return


        elif pos[2] >= self.canvas_width and self.x > 0:  # 패들의 동쪽이 600보다 크면
            return

        self.canvas.move(self.id, self.x, 0)

    def turn_left(self,evt):
        self.x = -9

    def turn_right(self,evt):
        self.x = 9

    # def turn_up(self, evt):
    #     self.y = -9
    #
    # def turn_down(self, evt):
    #     self.y = 9
tk = Tk()
tk.title("Game")   # 창에 글을 입력할 수 있다.
tk.resizable(0, 0)
tk.wm_attributes("-topmost", 1)
canvas = Canvas(tk, width=600, height=500, bd=0, highlightthickness=0)
canvas.pack()
tk.update()
paddle = Paddle(canvas,'white')             # 패들 색깔 지정
ball = Ball(canvas, paddle, 'white')         # 공 색깔 지정
start = False
                                                 #공을 약간 움직이고 새로운 위치로 화면을 다시 그리며, 잠깐 잠들었다가 다시 시작해라!
while 1:
    if ball.hit_bottom == False:
        ball.draw()
        paddle.draw()
        winner = ball.gameover()

    tk.update_idletasks()
    tk.update()
    print(winner)  # 그림을 다시 그려라! 라고 쉴새없이 명령
    if not start:
        time.sleep(0.2)   # 시작할때 2초간 멈췄다가 시작한다.
        start = True
    time.sleep(0.03)    # 공의 속도 조절