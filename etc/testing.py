from tkinter import *
import math
import time
import random

class Map() :
    def __init__(self, canvas):
        self.canvas = canvas
        self.canvas.create_polygon(0,   100,    0,      150, 700, 200, 700, 150)
        self.canvas.create_polygon(100, 300,    100,    350, 800, 300, 800, 250)
        self.canvas.create_polygon(0,   400,    0,      450, 700, 500, 700, 450)
        self.canvas.create_polygon(100, 600,    100,    650, 800, 600, 800, 550)

class Ball() :
    def __init__(self, canvas):
        self.canvas = canvas
        self.w_width = self.canvas.winfo_width()
        self.w_height = self.canvas.winfo_height()
        #self.id = self.canvas.create_oval(1,1,31,31)
        #self.x = 0
        #self.y = 3
        self.ball_list = []
        self.counter = 0
        self.random = None

    def reset(self):
        for i in self.ball_list :
            self.canvas.delete(i[0])
        self.ball_list = []
        self.counter = 0
        self.random = None

    def get_center(self, pos):
        return (round((pos[0]+pos[2])/2), pos[3])

    def draw(self):
        self.random = self.random or random.randrange(80, 200)
        # print(self.random)

        if (len(self.ball_list) < 5) and (self.counter % self.random == 0):
            self.ball_list.append([self.canvas.create_oval(1,1,25,25), 0, 3])
            self.counter = 0
            self.random = None
        self.counter += 1

        for j in range(len(self.ball_list)-1, -1, -1) :
            # 경사로 이동
            i = self.ball_list[j]
            id = i[0]

            pos = self.canvas.coords(i[0])
            base = self.get_center(pos)
            x, y = i[1], i[2]
            y=3
            if 550<=base[1]<=600 and base[0] >= 100 :
                if x == 0 : x = -3
                else : x -= 0.05
                y = math.ceil(-1*(base[0]-100)/14+600) - base[1]
            elif 250 <= base[1] <=300 and base[0] >= 100 :
                if x == 0 : x = -3
                else : x -= 0.05
                y = math.ceil(-1*(base[0]-100)/14+300) - base[1]
            elif 400<=base[1]<=450 and base[0] <= 700 :
                if x == 0 : x = 3
                else : x += 0.05
                y = math.ceil((base[0]/14)+400) - base[1]
            elif 100 <= base[1] <= 150 and base[0] <= 700:
                if x == 0 : x = 3
                else : x += 0.05
                y = math.ceil(base[0]/14+100) - base[1]
            else : y +=5

            if pos[0]+x <= 0 or pos[2]+x >= self.w_width :
                if x > 0:
                    self.canvas.move(id, self.w_width-pos[2]-2, y)
                elif x < 0:
                    self.canvas.move(id, -pos[0]+1, y)
                x = 0

            self.canvas.move(id, x, y)
            i[1], i[2] = x, y

            if pos[1] > self.w_height :
                self.canvas.delete(i[0])
                del self.ball_list[j]

class Man():

    dict = {}

    def __init__(self, canvas, ball):
        self.canvas = canvas
        self.w_width = self.canvas.winfo_width()
        self.w_height = self.canvas.winfo_height()
        self.id = self.canvas.create_rectangle(5, 550, 35, 599)
        self.goal = self.canvas.create_rectangle(450,75, 475, 100)
        self.x = 0
        self.y = 5
        self.ball = ball
        self.canvas.bind_all('<KeyPress-j>', self.switch)
        # self.canvas.bind_all('<KeyPress-l>', self.turn_right)
        # self.canvas.bind_all('<KeyPress-u>', self.jump_left)
        # self.canvas.bind_all('<KeyPress-o>', self.jump_right)
        self.val = False
        self.pre_pos = None
        self.jump = False
        self.move_list = [self.turn_left, self.turn_right, self.jump_left, self.jump_right]
        self.cycle_list = []

    def switch(self, evt):
        self.val = not self.val


    def play(self):
        print(Man.dict)
        if not self.jump :
            i = random.random()
            state = self.make_state()
            if i < 0.1 :
                move = self.move_random()
                state.append(move)
                state = tuple(state)
                if state not in Man.dict.keys() :
                    Man.dict[state] = 0
            else :
                state = self.move_learn(state)
                move = state[2]
            self.cycle_list.append(state)
            func = self.move_list[move]
            func()

    def move_random(self):
        val = random.randrange(0,4)
        return val

    def move_learn(self, state):
        temp_list = []
        for i in range(4) :
            temp = state[:]
            temp.append(i)
            temp = tuple(temp)
            if temp not in Man.dict.keys() :
                Man.dict[temp] = 0
            temp_list.append((temp, Man.dict[temp]))
        return max(temp_list, key = (lambda x : x[1]))[0]

    def make_state(self):
        ball_id = self.find_ball()
        ball_position = self.get_pos(ball_id)
        pos = self.get_pos()
        return [tuple(round(i/10) for i in pos), tuple(round(i/10) for i in ball_position)]

    def find_ball(self):
        if len(self.ball.ball_list) > 0 :
            ball_over = []
            pos = self.get_pos()
            for i in self.ball.ball_list :
                ball_pos = self.canvas.coords(i[0])
                if ball_pos[3] <= pos[1] :
                    ball_over.append((i[0], ball_pos[3]))
            return min(ball_over, key = (lambda x : x[1]))[0]


    def reset(self):
        self.canvas.delete(self.id)
        self.pre_pos = None
        self.jump = False
        self.id = self.canvas.create_rectangle(5, 550, 35, 599)
        self.cycle_list = []

    def get_pos(self, x = None):
        x = x or self.id
        pos = self.canvas.coords(x)
        return ((pos[0]+pos[2])/2, pos[3])

    def draw(self):
        base = self.get_pos()
        pos = self.canvas.coords(self.id)
        if not self.jump :
            self.pre_pos = None

        # 경사로 이동 & 점프 중
        if self.y == 5 :    # 경사로 이동
            if 525 <= base[1] <= 600 and base[0] >= 100 :
                if self.y + base[1] >= math.ceil(-1 * (base[0] - 100) / 14 + 600) :
                    self.y = math.ceil(-1 * (base[0] - 100) / 14 + 600) - base[1]
                    self.jump = False
            elif 225 <= base[1] <= 300 and base[0] >= 100 :
                if self.y + base[1] >= math.ceil(-1 * (base[0] - 100) / 14 + 300) :
                    self.y = math.ceil(-1 * (base[0] - 100) / 14 + 300) - base[1]
                    self.jump = False
            elif 375 <= base[1] <= 450 and base[0] <= 700 :
                if self.y + base[1] >= math.ceil((base[0] / 14) + 400) :
                    self.y = math.ceil((base[0] / 14) + 400) - base[1]
                    self.jump = False
            elif 75 <= base[1] <= 150 and base[0] <= 700 :
                if self.y + base[1] >= math.ceil(base[0] / 14 + 100) :
                    self.y = math.ceil(base[0] / 14 + 100) - base[1]
                    self.jump = False
            elif (0 <= base[0] <= 100) and 525 <= base[1] <= 600:
                if self.y + base[1] > 599 :
                    self.y = 599 - base[1]
                    self.jump = False
        else :  # 점프상태
            self.pre_pos = self.pre_pos or self.get_pos()
            if (base[1] == self.pre_pos[1] - 50) and self.y == -5 :
                self.y += 1
            elif (base[1] <= self.pre_pos[1] - 50) :
                self.y += 1

        # 벽
        if pos[0]+self.x < 0 or pos[2]+self.x > 800 :
            if self.x < 0 :
                self.x = pos[0]
            elif self.x > 0 :
                self.x = 799-pos[2]
        self.canvas.move(self.id, self.x, self.y)
        if not self.jump : self.y = 5

    def collision(self):
        pos = self.canvas.coords(self.id)
        goal_pos = self.canvas.coords(self.goal)
        for i in self.ball.ball_list :
            ball_pos = self.canvas.coords(i[0])
            if ball_pos[0] <= pos[2] and ball_pos[2] >= pos[0]:
                if ball_pos[1] <= pos[3] and ball_pos[3] >= pos[1] :
                    self.learning(-1)
                    self.reset()
                    self.ball.reset()
                    return
        if goal_pos[0] <= pos[2] and goal_pos[2] >= pos[0]:
            if goal_pos[1] <= pos[3] and goal_pos[3] >= pos[1]:
                self.learning(1)
                self.reset()
                self.ball.reset()
                return

    def learning(self, reward):
        self.cycle_list.reverse()
        for i in  self.cycle_list:
            pre_val = Man.dict[i]
            Man.dict[i] += 0.99*(reward - pre_val)
            reward *= 0.99


    def turn_left(self):
        if not self.jump:
            self.x = -4

    def turn_right(self):
        if not self.jump:
            self.x = 4

    def jump_left(self):
        if not self.jump:
            base = self.get_pos()
            self.jump = True
            if (((600 < base[0] < 700) and (base[1] > 550 or 250 < base[1] < 300))
                or ((100 < base[0] < 200) and (400 < base[1] < 450))):
                self.y = -15
            else:
                self.y = -5
            self.x = -4

    def jump_right(self):
        if not self.jump:
            base = self.get_pos()
            self.jump = True
            if (((600 < base[0] < 700) and (base[1] > 550 or 250 < base[1] < 300))
                or ((100 < base[0] < 200) and (400 < base[1] < 450))):
                self.y = -15
            else:
                self.y = -5
            self.x = 4

tk = Tk()
tk.title('Game')
tk.resizable(0, 0)

canvas = Canvas(tk, width = 800, height = 600, bd = 0, highlightthickness= 0)
canvas.pack()
tk.update()
map = Map(canvas)
ball = Ball(canvas)
man = Man(canvas, ball)
while True :
    man.play()
    ball.draw()
    man.draw()
    man.collision()
    tk.update_idletasks()
    tk.update()
    if man.val :
        time.sleep(0.02)