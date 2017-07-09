import numpy as np
import matplotlib.pyplot as plt

# plt.axis([0, 100, 0, 1])
plt.ion()

for i in np.arange(100):
    y = np.random.random()
    plt.plot(i,y)
    plt.pause(0.01)

while True:
    plt.pause(0.01)



import matplotlib.pyplot as plt
from drawnow import drawnow
import numpy as np

xList = [];yList = []

plt.ion()

def makeFig():
    plt.plot(xList, yList)

for i in range(50):
    y = np.random.random()
    xList.append(i)
    yList.append(y)
    drawnow(makeFig)
    plt.pause(0.001)







# 실시간으로 텍스트 파일 변경 정보 가져와서 출력(파일을 열고 계속 정보를 추가해도 그래프 생성)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style


fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
style.use('fivethirtyeight')

def animate(i):
    graph_data = open('e:\\data\\example.txt', 'r').read()
    print(graph_data)
    lines = graph_data.split('\n')
    xs = []
    ys = []

    for line in lines:
        if len(line) > 1:
            x, y = line.split(',')
            xs.append(x)
            ys.append(y)

        ax1.clear()
        ax1.plot(xs, ys)
        plt.pause(0.01)

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()









