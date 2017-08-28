from tkinter import *
import random

# define
WIDTH = 500
HEIGHT = 400
SEG_SIZE = 20
IN_GAME = True

class Snake:
    def __init__(self, segments):
        self.segments = segments
        self.mapping = {'Down':(0,1), 'Right':(1,0), 'Up':(0,-1), 'Left':(-1,0)}
        self.vector = self.mapping['Down']  # 시작 방향


    def move(self):
        for index in range(len(self.segments)-1):
            segment = self.segments[index].instance
            x1, y1, x2, y2 = canvas.coords(self.segments[index+1].instance)
            canvas.coords(segment, x1, y1, x2, y2)
            print(x1, y1, x2, y2)

        x1, y1, x2, y2 = canvas.coords(self.segments[-1].instance)
        canvas.coords(self.segments[-1].instance,
                 x1+self.vector[0]*SEG_SIZE, y1+self.vector[1]*SEG_SIZE,
                 x2+self.vector[0]*SEG_SIZE, y2+self.vector[1]*SEG_SIZE)

    # def add_segment(self):
    #     last_seg = canvas.coords(self.segments[0].instance)
    #     x = last_seg[2] - SEG_SIZE
    #     y = last_seg[3] - SEG_SIZE
    #     self.segments.insert(0, Segment(x,y))

    def change_direction(self, event):
        if event.keysym in self.mapping:
            self.vector = self.mapping[event.keysym]
            # print(self.vector)

    # def stop(self, evt):
    #

class Segment:
    def __init__(self,x,y):
        self.instance = canvas.create_rectangle(x, y, x+SEG_SIZE, y+SEG_SIZE, fill='white')


class Block1:
    def __init__(self):
        self.instance = canvas.create_rectangle(480, 0, 500, 360, fill='white')
        self.instance = canvas.create_rectangle(0, 0, 20, 400, fill='white')
        self.instance = canvas.create_rectangle(0, 380, 460, 400, fill='white')
        self.instance = canvas.create_rectangle(20, 0, 480, 20, fill='white')
#
# class Block2:
#     def __init__(self,x,y):
#         self.instance = canvas.create_rectangle(40, 20, x+SEG_SIZE, y+SEG_SIZE, fill='white')

# def create_block():
#     global BLOCK1, BLOCK2
#     # posx1 = SEG_SIZE * random.randint(1, (WIDTH-SEG_SIZE) / SEG_SIZE)
#     # posy1 = SEG_SIZE * random.randint(1, (HEIGHT-SEG_SIZE) / SEG_SIZE)
#     # posx2 = SEG_SIZE * random.randint(1, (WIDTH-SEG_SIZE) / SEG_SIZE)
#     # posy2 = SEG_SIZE * random.randint(1, (HEIGHT-SEG_SIZE) / SEG_SIZE)
#     BLOCK1 = canvas.create_rectangle(20, 40, 140, 60, fill='white')
#     BLOCK2 = canvas.create_rectangle(300, 80, 320, 200, fill='white')

# class Ball:
#     def __init__(self):
#         self.instance = canvas.create_oval(480, 380, 480+SEG_SIZE, 380+SEG_SIZE, fill='red')


# def create_ball():
#     global BALL1
#     posx1 = SEG_SIZE * random.randint(1, (WIDTH-SEG_SIZE) / SEG_SIZE)
#     posy1 = SEG_SIZE * random.randint(1, (HEIGHT-SEG_SIZE) / SEG_SIZE)
#     # posx2 = SEG_SIZE * random.randint(1, (WIDTH-SEG_SIZE) / SEG_SIZE)
#     # posy2 = SEG_SIZE * random.randint(1, (HEIGHT-SEG_SIZE) / SEG_SIZE)
#     BALL1 = canvas.create_oval(posx1, posy1, posx1+SEG_SIZE, posy1+SEG_SIZE, fill='red')
#     # BALL2 = canvas.create_oval(posx2, posy2, posx2+SEG_SIZE, posy2+SEG_SIZE, fill='yellow')


def main():
    global IN_GAME
    if IN_GAME:
        snake.move()
        head_coords = canvas.coords(snake.segments[-1].instance)
        x1, y1, x2, y2 = head_coords
        print([x1,y1,x2,y2])
        if x2 > WIDTH or x1 < 0 or y2 > HEIGHT or y1 < 0:
            IN_GAME = False
        # elif head_coords == canvas.coords(Ball):
            # snake.add_segment()
            # canvas.delete(Ball)
            # canvas.delete(BALL2)
            # canvas.delete(BLOCK1)
            # canvas.delete(BLOCK2)
            # Ball()
            # create_block()

        else:
            for index in range(len(snake.segments)-1):
                if head_coords == canvas.coords(snake.segments[index].instance):
                    IN_GAME = False
        tk.after(100, main)
    else:
        canvas.create_text(WIDTH/2, HEIGHT/2, text='WIN', font='Arial 20', fill='red')


if __name__ == '__main__':
    tk = Tk()
    tk.title('---지렁이 게임---')
    tk.resizable(0, 0)  # 게임창의 크기는 가로나 세로로 변경될수 없다라고 말하는것이다.
    tk.wm_attributes("-topmost", 1)  # 다른 모든 창들 앞에 캔버스를 가진 창이 위치할것을 tkinter 에게 알려준다.
    canvas = Canvas(tk, width=WIDTH, height=HEIGHT, bg='black', bd=0, highlightthickness=0)
    canvas.pack()
    # canvas.grid()
    canvas.focus_set()

    segments = [Segment(240, 200)]

    # segments = [Segment(SEG_SIZE, SEG_SIZE),
    #             Segment(SEG_SIZE*2, SEG_SIZE),
    #             Segment(SEG_SIZE*3, SEG_SIZE),
    #             Segment(SEG_SIZE*4, SEG_SIZE)]

    snake = Snake(segments)
    canvas.bind('<KeyPress>', snake.change_direction)
    # create_block()
    Block1()
    # Ball()
    # Block2(200,20)
    main()
    tk.mainloop()











# from tkinter import*
#
# class MouseKeyEventDemo:
#     def __init__(self):
#         window = Tk()
#         window.title("마우스 위치")
#         self.canvas = Canvas(window, bg="white", width=800, height=600)
#         self.canvas.pack()
#
#         self.canvas.bind("<Button-1>", self.processMouseEvent)
#
#         self.canvas.focus_set()
#
#         window.mainloop()
#
#     def processMouseEvent(self,event):
#         self.canvas.create_text(event.x, event.y, text=(event.x,event.y))
#
#
#
# MouseKeyEventDemo()
