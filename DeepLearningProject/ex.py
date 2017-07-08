# import numpy as np
# import matplotlib.pyplot as plt
#
# plt.axis([0, 100, 0, 1])
# plt.ion()
#
# for i in range(100):
#     y = np.random.random()
#
#     # plt.plot(i,y)
#     plt.scatter(i, y)
#     plt.pause(0.01)
#
# while True:
#     plt.pause(0.01)



import matplotlib.pyplot as plt
from drawnow import drawnow
import numpy as np

xList = [];yList = []

plt.ion()
# fig = plt.figure()

def makeFig():
    plt.plot(xList, yList)

for i in np.arange(50):
    y = np.random.random()
    xList.append(i)
    yList.append(y)
    drawnow(makeFig)
    plt.pause(0.001)








import matplotlib.pyplot as plt
import matplotlib.animation as animation


fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)


def animate(i):
    graph_data = open('e:\\data\\final_incheon_airport.txt', 'r').read()
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


# -- 실시간으로 텍스트 파일 변경 정보 가져와서 출력함

#    파일을 열고 계속 정보를 추가하면 그래프가 움직임.

ani = animation.FuncAnimation(fig, animate, interval=1)

plt.show()








