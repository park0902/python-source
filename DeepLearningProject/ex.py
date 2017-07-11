# import numpy as np
# import matplotlib.pyplot as plt
#
# # plt.axis([0, 100, 0, 1])
# plt.ion()
#
# for i in np.arange(100):
#     y = np.random.random()
#     plt.scatter(i,y)
#     plt.pause(0.01)
#
# while True:
#     plt.pause(0.01)



import matplotlib.pyplot as plt
from drawnow import drawnow
import numpy as np

xList = [];yList = []

plt.ion()

def makeFig1():
    plt.subplot(1, 5, 1)
    plt.plot(xList, yList)
    plt.title('Batch_size & Cost')
    plt.grid(True)
    plt.ylabel('Cost')
    plt.xlabel('Batch_size')
    plt.legend(loc='upper right')

def makeFig2():
    plt.subplot(1, 5, 2)
    plt.plot(xList, yList)
    plt.title('Batch_size & Cost')
    plt.grid(True)
    plt.ylabel('Cost')
    plt.xlabel('Batch_size')
    plt.legend(loc='upper right')

def makeFig3():
    plt.subplot(1, 5, 3)
    plt.plot(xList, yList)
    plt.title('Batch_size & Cost')
    plt.grid(True)
    plt.ylabel('Cost')
    plt.xlabel('Batch_size')
    plt.legend(loc='upper right')

def makeFig4():
    plt.subplot(1, 5, 4)
    plt.plot(xList, yList)
    plt.title('Batch_size & Cost')
    plt.grid(True)
    plt.ylabel('Cost')
    plt.xlabel('Batch_size')
    plt.legend(loc='upper right')

def makeFig5():
    plt.subplot(1, 5, 5)
    plt.plot(xList, yList)
    plt.title('Batch_size & Cost')
    plt.grid(True)
    plt.ylabel('Cost')
    plt.xlabel('Batch_size')
    plt.legend(loc='upper right')

for j in range(100):
    y = np.random.random()
    xList.append(j)
    yList.append(y)
    drawnow(makeFig1)
    if j >= 50:
        y = np.random.random()
        xList.append(j)
        yList.append(y)
        drawnow(makeFig2)
        # print(j,y)


    # plt.subplot(1, 10, j + 1)
    # drawnow(makeFig1)

    # drawnow(makeFig3)
    # drawnow(makeFig4)
    # drawnow(makeFig5)
    # plt.plot(xList,yList)
    print(j+1, y)
    if j == 30:
        plt.subplot(1,2,2)
        y = np.random.random()
        xList.append(j)
        yList.append(y)
        # plt.subplot(1, 10, j + 1)
        drawnow(plt.plot(xList, yList))


    # print(i,y)
    #
    plt.pause(0.001)
    if j == 5:
        y = np.random.random()
        xList.append(j)
        yList.append(y)
        print(j + 1, y)
        # print(i,y)
        plt.plot(xList,yList)
        plt.pause(0.001)









# # 실시간으로 텍스트 파일 변경 정보 가져와서 출력(파일을 열고 계속 정보를 추가해도 그래프 생성)
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from matplotlib import style
#
#
# fig = plt.figure()
# ax1 = fig.add_subplot(1, 1, 1)
# style.use('fivethirtyeight')
#
# def animate(i):
#     graph_data = open('d:\\data\\example.txt', 'r').read()
#     print(graph_data)
#     lines = graph_data.split('\n')
#     xs = []
#     ys = []
#
#     for line in lines:
#         if len(line) > 1:
#             x, y = line.split(',')
#             xs.append(x)
#             ys.append(y)
#
#         ax1.clear()
#         ax1.plot(xs, ys)
#         plt.pause(0.01)
#
# ani = animation.FuncAnimation(fig, animate, interval=1000)
# plt.show()









