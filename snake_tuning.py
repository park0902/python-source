from tkinter import *
import random

# define
WIDTH = 1000
HEIGHT = 800
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

        x1, y1, x2, y2 = canvas.coords(self.segments[-2].instance)
        canvas.coords(self.segments[-1].instance,
                 x1+self.vector[0]*SEG_SIZE, y1+self.vector[1]*SEG_SIZE,
                 x2+self.vector[0]*SEG_SIZE, y2+self.vector[1]*SEG_SIZE)

    def add_segment(self):
        last_seg = canvas.coords(self.segments[0].instance)
        x = last_seg[2] - SEG_SIZE
        y = last_seg[3] - SEG_SIZE
        self.segments.insert(0, Segment(x,y))

    def change_direction(self, event):
        if event.keysym in self.mapping:
            self.vector = self.mapping[event.keysym]
            # print(self.vector)


class Segment(object):
    def __init__(self,x,y):
        self.instance = canvas.create_rectangle(x, y, x+SEG_SIZE, y+SEG_SIZE, fill='white')


def create_ball():
    global BALL1,BALL2
    posx1 = SEG_SIZE * random.randint(1, (WIDTH-SEG_SIZE) / SEG_SIZE)
    posy1 = SEG_SIZE * random.randint(1, (HEIGHT-SEG_SIZE) / SEG_SIZE)
    posx2 = SEG_SIZE * random.randint(1, (WIDTH-SEG_SIZE) / SEG_SIZE)
    posy2 = SEG_SIZE * random.randint(1, (HEIGHT-SEG_SIZE) / SEG_SIZE)
    BALL1 = canvas.create_oval(posx1, posy1, posx1+SEG_SIZE, posy1+SEG_SIZE, fill='red')
    BALL2 = canvas.create_oval(posx2, posy2, posx2+SEG_SIZE, posy2+SEG_SIZE, fill='yellow')


def main():
    global IN_GAME
    if IN_GAME:
        snake.move()
        head_coords = canvas.coords(snake.segments[-1].instance)
        x1, y1, x2, y2 = head_coords
        # print([x1,y1,x2,y2])
        if x2 > WIDTH or x1 < 0 or y2 > HEIGHT or y1 < 0:
            IN_GAME = False
        elif head_coords == canvas.coords(BALL1) or head_coords == canvas.coords(BALL2):
            snake.add_segment()
            canvas.delete(BALL1)
            canvas.delete(BALL2)
            create_ball()

        else:
            for index in range(len(snake.segments)-1):
                if head_coords == canvas.coords(snake.segments[index].instance):
                    IN_GAME = False
        tk.after(100, main)
    else:
        canvas.create_text(WIDTH/2, HEIGHT/2, text='GAME OVER', font='Arial 20', fill='red')


if __name__ == '__main__':
    tk = Tk()
    tk.title('---지렁이 게임---')
    tk.resizable(0, 0)  # 게임창의 크기는 가로나 세로로 변경될수 없다라고 말하는것이다.
    tk.wm_attributes("-topmost", 1)  # 다른 모든 창들 앞에 캔버스를 가진 창이 위치할것을 tkinter 에게 알려준다.
    canvas = Canvas(tk, width=WIDTH, height=HEIGHT, bg='black', bd=0, highlightthickness=0)
    canvas.pack()
    # canvas.grid()
    canvas.focus_set()

# segments = [Segment(SEG_SIZE, SEG_SIZE),
#             Segment(SEG_SIZE*2, SEG_SIZE),
#             Segment(SEG_SIZE*3, SEG_SIZE)]

    segments = [Segment(SEG_SIZE, SEG_SIZE),
                Segment(SEG_SIZE*2, SEG_SIZE),
                Segment(SEG_SIZE*3, SEG_SIZE),
                Segment(SEG_SIZE*4, SEG_SIZE)]

    snake = Snake(segments)
    canvas.bind('<KeyPress>', snake.change_direction)
    create_ball()
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
