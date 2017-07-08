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

def makeFig():
    plt.scatter(xList,yList) # I think you meant this

plt.ion() # enable interactivity
fig=plt.figure() # make a figure

xList=list()
yList=list()

for i in np.arange(50):
    y=np.random.random()
    xList.append(i)
    yList.append(y)
    drawnow(makeFig)
    #makeFig()      The drawnow(makeFig) command can be replaced
    #plt.draw()     with makeFig(); plt.draw()
    plt.pause(0.001)