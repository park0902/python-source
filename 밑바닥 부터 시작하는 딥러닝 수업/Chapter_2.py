# w1 * x1 + w2 * x2 파이썬으로 구현
import numpy as np

x = np.array([0,1])
w = np.array([0.5])

print(np.sum(x*w))



# 편향을 더해서 파이썬으로 구현
import numpy as np

x = np.array([0,1])
w = np.array([0.5])
b = -0.7

print(np.sum(x*w) + b)



# AND 함수를 파이썬으로 구현
def AND(x1, x2):
    w1, w2, b = 0.5, 0.5, -0.7
    re = x1*w1 + x2*w2 + b

    if re <= 0:
        return 0
    else:
        return 1

print(AND(0,0))
print(AND(1,1))



# NAND 함수를 파이썬으로 구현
import numpy as np

def NAND(x1, x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.7
    re = np.sum(x*w) + b

    if re <= 0:
        return 0
    else:
        return 1

print(NAND(0,0))
print(NAND(1,1))



# OR 함수를 파이썬으로 구현
import numpy as np

def OR(x1, x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.2
    re = np.sum(x*w) + b

    if re <= 0:
        return 0
    else:
        return 1

print(OR(0,0))
print(OR(1,1))



# XOR 함수를 파이썬으로 구현
import numpy as np

def XOR(x1,x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y = AND(s1,s2)

    return y

print(XOR(1,1))








import numpy as np

x = np.array([1,2])
y = np.array([3,4])

print(2*x + y)




import numpy as np

def andPerceptron(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    netInput = x1*w1 + x2*w2
    if netInput <= theta:
        return 0
    elif netInput > theta:
        return 1

def nandPerceptron(x1, x2):
    w1, w2, theta = -0.5, -0.5, -0.7
    netInput = x1*w1 + x2*w2
    if netInput <= theta:
        return 0
    elif netInput > theta:
        return 1

def orPerceptron(x1, x2):
    w1, w2, bias = 0.5, 0.5, -0.2
    netInput = x1*w1 + x2*w2 + bias
    if netInput <= 0:
        return 0
    else:
        return 1


def xorPerceptron(x1, x2):
    s1 = nandPerceptron(x1,x2)
    s2 = orPerceptron(x1,x2)
    y = andPerceptron(s1,s2)

    return y




inputData = np.array([[0,0],[0,1],[1,0],[1,1]])

print("---And Perceptron---")
for xs1 in inputData:
    print(str(xs1) + " ==> " + str(andPerceptron(xs1[0], xs1[1])))

print("---Nand Perceptron---")
for xs2 in inputData:
    print(str(xs2) + " ==> " + str(nandPerceptron(xs2[0], xs2[1])))

print("---Or Perceptron---")
for xs3 in inputData:
    print(str(xs3) + " ==> " + str(orPerceptron(xs3[0], xs3[1])))

print("---Xor Perceptron---")
for xs4 in inputData:
    print(str(xs4) + " ==> " + str(xorPerceptron(xs4[0], xs4[1])))