from tkinter import *
import random

# Globals
WIDTH = 200
HEIGHT = 120
SEG_SIZE = 10
IN_GAME = True


class Snake(object):
    def __init__(self, segments):
        self.segments = segments
        self.mapping = [1,2,3,4]
        random.shuffle(self.mapping)
        self.vector = self.mapping[0]
        # self.mapping = {'Down':(0,1), 'Right':(1,0), 'Up':(0,-1), 'Left':(-1,0)}
        # self.vector = self.mapping['Right']

    def move(self):
        for index in range(len(self.segments)-1):
            segment = self.segments[index].instance
            x1, y1, x2, y2 = canvas.coords(self.segments[index+1].instance)
            canvas.coords(segment, x1, y1, x2, y2)

        x1, y1, x2, y2 = canvas.coords(self.segments[-1].instance)
        canvas.coords(self.segments[-1].instance,
                 x1+self.vector*SEG_SIZE, y1+self.vector*SEG_SIZE,
                 x2+self.vector*SEG_SIZE, y2+self.vector*SEG_SIZE)

    # def add_segment(self):
    #     last_seg = c.coords(self.segments[0].instance)
    #     x = last_seg[2] - SEG_SIZE
    #     y = last_seg[3] - SEG_SIZE
    #     self.segments.insert(0, Segment(x,y))

    def change_direction(self, event):
        if event.keysym in self.mapping:
            self.vector = self.mapping[event.keysym]


class Segment(object):
    def __init__(self, x, y):
        self.instance = canvas.create_rectangle(x, y, x + SEG_SIZE, y + SEG_SIZE, fill='white')



# class Block1:
#     def __init__(self,x,y):
#         self.instance = canvas.create_rectangle(40, 20, x+SEG_SIZE, y+SEG_SIZE, fill='white')
#
# class Block2:
#     def __init__(self,x,y):
#         self.instance = canvas.create_rectangle(40, 20, x+SEG_SIZE, y+SEG_SIZE, fill='white')

# def create_block():
#     global BLOCK1, BLOCK2
    # posx1 = SEG_SIZE * random.randint(1, (WIDTH-SEG_SIZE) / SEG_SIZE)
    # posy1 = SEG_SIZE * random.randint(1, (HEIGHT-SEG_SIZE) / SEG_SIZE)
    # posx2 = SEG_SIZE * random.randint(1, (WIDTH-SEG_SIZE) / SEG_SIZE)
    # posy2 = SEG_SIZE * random.randint(1, (HEIGHT-SEG_SIZE) / SEG_SIZE)
    # BLOCK1 = canvas.create_rectangle(20, 40, 140, 60, fill='white')
    # BLOCK2 = canvas.create_rectangle(300, 80, 320, 200, fill='white')


# Helper function
def create_ball():
    global BLOCK
    # posx = SEG_SIZE * random.randint(1, (WIDTH-SEG_SIZE) / SEG_SIZE)
    # posy = SEG_SIZE * random.randint(1, (HEIGHT-SEG_SIZE) / SEG_SIZE)
    BLOCK = canvas.create_oval(190, 110, 200, 120, fill='red')

def main():
    global IN_GAME
    if IN_GAME:
        snake.move()
        head_coords = canvas.coords(snake.segments[-1].instance)
        x1, y1, x2, y2 = head_coords
        if x2 > WIDTH or x1 < 0 or y2 > HEIGHT or y1 < 0:
            IN_GAME = False
        elif head_coords == canvas.coords(BLOCK):
            # s.add_segment()
            canvas.delete(BLOCK)
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

    segments = [Segment(SEG_SIZE, SEG_SIZE)]

    # segments = [Segment(SEG_SIZE, SEG_SIZE),
    #             Segment(SEG_SIZE*2, SEG_SIZE),
    #             Segment(SEG_SIZE*3, SEG_SIZE),
    #             Segment(SEG_SIZE*4, SEG_SIZE)]

    snake = Snake(segments)
    canvas.bind('<KeyPress>', snake.change_direction)
    create_ball()
    # create_block()
    # Block1(40,200)
    # Block2(200,20)
    main()
    tk.mainloop()