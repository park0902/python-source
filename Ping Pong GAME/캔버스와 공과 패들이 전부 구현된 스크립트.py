from tkinter import *
import random
import time

class Ball:

    def __init__(self, canvas, paddle, color):
        self.canvas = canvas
        self.paddle = paddle
        self.id = canvas.create_oval(10, 10, 25, 25, fill=color)
        #                            서   남  동  북
        canvas.configure(background='black')    # 캔버스의 색깔을 검정색으로 하겠다
        self.canvas.move(self.id, 245, 100)
        #                   공     x축  y축

        starts = [-3,-2,-1, 1, 2, 3]      # 음수는 왼쪽방향 양수는 오른쪽 방향 0은 수직 방향
        random.shuffle(starts)

        self.x = starts[0]
        self.y = -3
        # self.x = 0
        # self.y = 0
        self.canvas_height = self.canvas.winfo_height() # 캔버스의 높이
        # print('높이 : ', self.canvas_height)
        self.canvas_width = self.canvas.winfo_width()   # 캔버스의 넓이
        # print('넓이 : ', self.canvas_width)
        self.hit_bottom = False                         # 바닥의 공이 닿으면 게임이 끝나는 코드를
                                                        # 구현하기 위해서 쓰는 변수
        self.isMiss = False
        # self.canvas.bind_all('<Key>', self.turn_up)
        # self.canvas.bind_all('<KeyPress-Left>',self.turn_left)      # 키보드 방향키 <- 를 누르면 turn_left 함수 실행
        # self.canvas.bind_all('<KeyPress-Right>',self.turn_right)    # 키보드 방향키 -> 를 누르면 turn_right 함수 실행
        # self.canvas.bind_all('<KeyPress-Up>',self.turn_up)          # 키보드 방향키 ^ 를 누르면 turn_up 함수 실행
        # self.canvas.bind_all('<KeyPress-Down>',self.turn_down)      # 키보드 방향키  를 누르면 turn_down 함수 실행

    # def turn_up(self,evt):
    #     self.x = random.sample(range(-5,6),1)
    #     self.y = 5
    # def turn_left(self, evt):
    #     self.x = -9
    #
    # def turn_right(self, evt):
    #     self.x = 9
    #
    # def turn_up(self,evt):
    #     self.y = -9
    #
    # def turn_down(self,evt):
    #     self.y = 9

    # def turn_up(self, evt):
    #     starts1 = [-3,-2,-1]
    #     starts2 = [-3, -2, -1, 1, 2, 3]     # 음수는 왼쪽방향 양수는 오른쪽 방향 0은 수직 방향
    #     random.shuffle(starts1)
    #     random.shuffle(starts2)
    #     self.y = starts1[0]
    #     self.x = starts2[0]

    def draw(self):
        self.canvas.move(self.id, self.x, self.y)
        #                   공  공의 x방향 공의 y방향
        #                        (좌 우)  (위 아래)
        pos = self.canvas.coords(self.id)
        # print(pos)
        '''  서쪽     남쪽    동쪽    북쪽  좌표
            [258.0, 107.0, 273.0, 122.0]
         pos   0      1     2       3
        '''
        if pos[1] <= 0:
            self.y = 3
        if pos[3] >= self.canvas_height:
            # self.hit_bottom = True
            self.y =-3
            # print('miss')
        if pos[0] <= 0:
            self.x = 3
        if pos[2] >= self.canvas_width:
            self.x = -3
        if self.hit_paddle(pos) == True:
            self.y = -3
            # self.x = 0
            # self.x = paddle.x
            # print('hit')


    def gameover(self):
#         pos = self.canvas.coords(self.id)
#         if pos[1] < 400 < pos[3] and self.isMiss == False and self.y >0:
# #          공 남쪽         공 북쪽                공의 방향이 아래로 향해있다면
#             self.isMiss = True  # 공이 400을 지나갈때 miss 가 여러번 출력되기때문에
#             # print('miss')       # 한 번만 출력되게 할려고 isMiss 변수 사용!
#         elif pos[1] > 400 or pos[3] < 400:
#             self.isMiss = False

        pos = self.canvas.coords(self.id)
        paddle_pos = self.canvas.coords(self.paddle.id)
        if pos[3] == paddle_pos[3] and self.y > 0:
            return -1
        elif self.hit_paddle(pos) ==  True:
            return 1
        else:
            return 0
        return

    def hit_paddle(self,pos):
        paddle_pos = self.canvas.coords(self.paddle.id)
        if pos[2] >= paddle_pos[0] and pos[0] <= paddle_pos[2]:
#       만약 공의 동쪽 >= 패들의 서쪽 and 공의 서쪽 <= 패들의 동쪽 이면
            if pos[3] >= paddle_pos[1] and pos[1] <= paddle_pos[3]:
#           만약 공의 북쪽 >= 패들의 남쪽 and 공의 남쪽 <= 패들의 북쪽 이면
                return True
        return False

    def winner(self, winner):
        return winner

class Paddle:

    def __init__(self,canvas,color):
        self.canvas = canvas
        self.id = canvas.create_rectangle(0,0,100,10,fill=color)    # 패들의 크기와 색깔
        self.canvas.move(self.id, 200, 400)                         # 패들을 움직이게 하는 함수
                                                                    # x축 200의 y축 400 쪽에 패들을 고정시키는 코드
        self.x = 0                                                  # 패들이 게임 시작할때 움직이지 말라고 속도 고정
        self.canvas_width = self.canvas.winfo_width()               # 패들이 화면 밖으로 나가지 않도록 하는 코드
        self.canvas.bind_all('<KeyPress-Left>',self.turn_left)      # 키보드 방향키 <- 를 누르면 turn_left 함수 실행
        self.canvas.bind_all('<KeyPress-Right>',self.turn_right)    # 키보드 방향키 -> 를 누르면 turn_right 함수 실행
        # self.canvas.bind_all('<KeyPress-Up>',self.turn_up)          # 키보드 방향키 ^ 를 누르면 turn_up 함수 실행
        # self.canvas.bind_all('<KeyPress-Down>',self.turn_down)      # 키보드 방향키  를 누르면 turn_down 함수 실행

    def draw(self):
        # self.canvas.move(self.id, self.x, 0)
        pos = self.canvas.coords(self.id)
        if pos[0] <= 0 and self.x < 0:                     # 패들 서쪽이 0보다 작다면
            # self.x = 0                                   # 패들을 멈춰라
            return                                         # 함수 종료!
        elif pos[2] >= self.canvas_width and self.x > 0:   # 패들의 동쪽이 600보다 크다면
            # self.x = 0                                   # 패들을 멈춰라
            return                                         # 함수 종료!

        self.canvas.move(self.id, self.x, 0)

    def turn_left(self,evt):
        self.x = -9

    def turn_right(self,evt):
        self.x = 9

    # def turn_up(self,evt):
    #     self.y = -9
    #     self.canvas.move(self.id, self.x, self.y)
    #
    # def turn_down(self,evt):
    #     self.canvas.move(self.id, self.x, self.y)
    #     self.y = 9

tk = Tk()
tk.title("Game")
tk.resizable(0, 0)
tk.wm_attributes("-topmost", 1)
canvas = Canvas(tk, width=600, height=500, bd=0, highlightthickness=0)
canvas.pack()
tk.update()
paddle = Paddle(canvas,'white')
ball = Ball(canvas, paddle, 'white')
start = False
while 1:
    if ball.hit_bottom == False:
        ball.draw()
        paddle.draw()
        winner = ball.gameover()
    # ball.draw()             # ball 인스턴스의 draw 메소드 실행
    #                         # (공이 벽에 부딪혀도 화면 밖으로 나가지 않도록)
    # paddle.draw()           # paddle 인스턴스의 draw 메소드 실행
    #                         # (패들을 키보드로 조정하면서 화면 밖으로 나가지 않도록)
    # ball.gameover()         # ball 인스턴스의 gameover 메소드 실행


    tk.update_idletasks()   # tkinter 에게 계속 화면을 그리라고 명령
    tk.update()             # 구현된 내용 반영
    print(winner)  # 그림을 다시 그려라! 라고 쉴새없이 명령

    if not start:
        time.sleep(0.2)  # 시작할때 2초간 멈췄다가 시작한다.
        start = True
    time.sleep(0.02)  # 게임을 사람이 보기 편하게 100의 2초마다 잠들어라!
    # (실제로 학습할때는 이 부분을 빼고 수행해야 빨리 학습할 수 있다)