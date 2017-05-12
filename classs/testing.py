from tkinter import *
import random
import time

class Ball:

    def __init__(self, canvas, paddle, color):
        self.canvas = canvas
        self.paddle = paddle
        self.id = canvas.create_oval(10, 10, 25, 25, fill=color)    # 공의 크기(좌상단, 우하단), 색상 정의
        canvas.configure(background='black')
        self.canvas.move(self.id, 245,100) # 공의 초기 위치

        starts = [-3, -2, -1, 1, 2, 3]  # 공의 초기 속도 결정. x 방향 속도를 무작위로 선택.
        random.shuffle(starts)
        self.x = starts[0]
        self.y = -3 # y 방향 속도 고정.

        self.canvas_height = self.canvas.winfo_height() # 캔버스의 크기를 받음
        self.canvas_width = self.canvas.winfo_width()

        self.hit_bottom = False # 공과 바닥의 충돌 여부를 결정하는 함수



    def draw(self):
        self.canvas.move(self.id, self.x, self.y)

        pos = self.canvas.coords(self.id)   # 공의 좌표(좌상단, 우하단)
                                            # [507.0, 140.0, 522.0, 155.0]
        if pos[1] <= 0: # 천장에 부딪히면 아랫방향으로 튀김
            self.y = 3
        if pos[3] >= self.canvas_height:    # 바닥에 부딪히면 상측으로 튕겨야함
            # self.hit_bottom = True
            self.y = -3
        if pos[0] <= 0: # 좌측벽에 부딪히면 우측으로 튕김
            self.x = 3
        if pos[2] >= self.canvas_width: # 우측벽에 부딪히면 좌측으로 튕김
            self.x = -3
        if self.hit_paddle(pos) == True:    # 패들에 부딪히면 상측으로 튕김
            self.y = -3

    def hit_paddle(self,pos):   # 패들의 위치와 공의 위치를 비교하여 튕기게 함
        paddle_pos = self.canvas.coords(self.paddle.id)
        if pos[2] >= paddle_pos[0] and pos[0] <= paddle_pos[2]:
            if pos[3] >= paddle_pos[1] and pos[1] <= paddle_pos[3]:
                return True
        return False

    def gameover(self):
        pos = self.canvas.coords(self.id)
        if pos[1] < 400 < pos[3] and self.isMiss == False and self.y > 0:
            self.isMiss = True
            return 'miss'
        elif pos[1] > 400 or pos[3] < 400:
            self.isMiss = False
        elif self.hit_paddle(pos):
            return 'hit'

        return 'pass'


class Paddle:

    def __init__(self,canvas,color):
        self.canvas = canvas
        self.id = canvas.create_rectangle(0,0,100,10,fill=color)    # 패들의 크기, 색상
        self.canvas.move(self.id, 200, 400) # 패들의 초기 위치
        self.x = 0  # 패들의 초기속도. 정지상태로 시작
        self.canvas_width = self.canvas.winfo_width()   # 패들의 운동범위
        self.x_range = 600 - 100
        self.ball = None
        self.check_ball = False # 공이 측정선을 넘었는가
        self.ball_config = None # 공위치, 속도
        self.is_moving = False  # 페들이 목적지로 이동중인가
        self.dest = None    # 페들의 목적지
        self.values={}

    def check(self):
        pos = self.canvas.coords(self.ball.id)
        if pos[1] < 100 and self.ball.y < 0 :
            self.check_ball = True
            self.ball_config = (tuple(pos), self.ball.x, self.ball.y)

    def randomchoice(self):
        return random.randrange(0, self.x_range)

    def greedy(self):
        if self.ball_config not in self.values :
            return self.randomchoice()
        return self.values[self.ball_config][0]


    def add(self, val):
        pos = self.canvas.coords(self.id)
        if self.ball_config not in self.values:
            self.values[self.ball_config] = [pos[0], val]
        else :
            if val > self.values[self.ball_config][1] :
                print('갱신')
                self.values[self.ball_config] = [pos[0], val]

    def get_val(self):
        ball_pos = self.canvas.coords(self.ball.id)
        paddle_pos = self.canvas.coords(self.id)
        ball_center = (ball_pos[0]+ball_pos[2])/2
        center = (paddle_pos[0]+paddle_pos[2])/2
        if ball_center >= center :
            result = abs(paddle_pos[2]-ball_pos[0])
        else :
            result = abs(paddle_pos[0]-ball_pos[2])
        return result

    def reset(self):
        self.x = 0  # 패들의 초기속도. 정지상태로 시작
        self.canvas_width = self.canvas.winfo_width()  # 패들의 운동범위
        self.check_ball = False  # 공이 측정선을 넘었는가
        self.ball_config = None  # 공위치, 속도
        self.is_moving = False  # 페들이 목적지로 이동중인가
        self.dest = None  # 페들의 목적지

        self.canvas.delete(self.id)
        self.id = canvas.create_rectangle(0, 0, 100, 10, fill='white')  # 패들의 크기, 색상
        self.canvas.move(self.id, 200, 400)  # 패들의 초기 위치
        self.canvas.delete(self.ball.id)
        self.ball.id = canvas.create_oval(10, 10, 25, 25, fill='white')  # 공의 크기(좌상단, 우하단), 색상 정의
        self.canvas.move(self.ball.id, 245, 100)  # 공의 초기 위치

        starts = [-3, -2, -1, 1, 2, 3]  # 공의 초기 속도 결정. x 방향 속도를 무작위로 선택.
        random.shuffle(starts)
        self.ball.x = starts[0]
        self.ball.y = -3  # y 방향 속도 고정.

    def draw(self):
        pos = self.canvas.coords(self.id)
        result = self.ball.gameover()
        if result == 'hit' :
            self.add(self.get_val())
            self.reset()
            return
        elif result == 'miss' :
            self.reset()
            return

        if not self.check_ball :
            self.check()
        elif not self.is_moving :
            r = random.random()
            if r < 0.1 :
                self.dest = self.randomchoice()
            else :
                self.dest = self.greedy()
            self.is_moving = True
        else :
            if self.dest > pos[0] :
                self.canvas.move(self.id, 3, 0)
            elif self.dest < pos[0] :
                self.canvas.move(self.id, -3, 0)

    def turn_left(self, evt):
        self.x = -9

    def turn_right(self, evt):
        self.x = 9


tk = Tk()
tk.title("Game")
tk.resizable(0, 0)
tk.wm_attributes("-topmost", 1)
canvas = Canvas(tk, width=600, height=500, bd=0, highlightthickness=0)
canvas.pack()
tk.update()
paddle = Paddle(canvas,'white')
ball = Ball(canvas, paddle, 'white')
paddle.ball = ball


while 1:

    ball.draw()             # 공 인스턴스의 draw 메소드 실행. 벽과 패들에 튕김.
    paddle.draw()           # 패들을 키보드로 조종하며 화면 밖으로 나가지 않음.
    tk.update_idletasks()   # tkinter 가 화면을 계속 새로 그리도록 명령.
    tk.update()             # 구현된 내용들을 갱신.
    time.sleep(0.005)        # 0.02초 대기. 기계학습시에는 제한을 풀고 빠르게 수행한다.