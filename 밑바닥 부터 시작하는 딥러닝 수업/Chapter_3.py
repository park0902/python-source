# 계단함수를 파이썬으로 구현
import numpy as np

def step_function(x):
    y = x > 0

    return y.astype(np.int)     # true 는 1 로 변경, false 는 0 으로 변경

x_data = np.array([-1,0,1])

print(step_function(x_data))



# 계단함수 그래프 그리기
import numpy as np
import matplotlib.pyplot as plt

def step_function(x):

    return np.array(x > 0, dtype=np.int)

x_data = np.arange(-5.0,5.0,0.1)
y_data = step_function(x_data)

plt.plot(x_data, y_data)
plt.ylim(-0.1,1.1)
plt.show()





# 계단함수를 이용하여 파이썬으로 구현
import numpy as np

def step_function(x):
    y = x > 0

    return y.astype(np.int)     # true 는 1 로 변경, false 는 0 으로 변경

x_data = np.array([-1,0,0])
w_data = np.array([0.3,0.4,0.1])
o_data = np.sum(x_data * w_data)

print(o_data)
print(step_function(o_data))



# 시그모이드 함수를 파이썬으로 구현
import numpy as np

def sigmoid(x):

    return 1 / (1 + np.exp(-x))

print(sigmoid(1.0))
print(sigmoid(2.0))



# 시그모이드 함수 그래프 그리기
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):

    return 1 / (1 + np.exp(-x))

x_data = np.arange(-5.0, 5.0, 0.1)
y_data = sigmoid(x_data)

plt.plot(x_data, y_data)
plt.ylim(-0.1, 1.1)
plt.show()


# 시그모이드 함수를 반대로 그래프 그리기
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(x))

x_data = np.arange(-5.0, 5.0, 0.1)
y_data = sigmoid(x_data)

plt.plot(x_data, y_data)
plt.ylim(-0.1, 1.1)
plt.show()



# 계단함수와 시그모이드 함수 같이 그리기
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):

    return 1 / (1 + np.exp(-x))

def step_function(x):

    return np.array(x > 0, dtype=np.int)


x_data = np.arange(-5.0, 5.0, 0.1)
y_data1 = sigmoid(x_data)
y_data2 = step_function(x_data)

plt.plot(x_data, y_data1)
plt.plot(x_data, y_data2, ls='--')
plt.xlim(-6,6);plt.ylim(-0.1, 1.1)
plt.show()




# ReLu 함수를 파이썬으로 구현
import numpy as np

def relu(x):

    return np.maximum(0,x)


print(relu(-1))
print(relu(0.3))



# ReLu 함수 그래프 그리기
import numpy as np
import matplotlib.pyplot as plt

def relu(x):

    return np.maximum(0,x)


x_data = np.arange(-5.0, 5.0, 0.1)
y_data = relu(x_data)

plt.plot(x_data, y_data)
plt.show()



# 행렬의 내적을 파이썬으로 구현
import numpy as np

a = np.array([[1,2,3], [4,5,6]])
b = np.array([[5,6], [7,8], [9,10]])

print(np.dot(a,b))



# 행렬의 곱을 파이썬으로 구현
import numpy as np

a = np.array([[5,6], [7,8], [9,10]])
b = np.array([[1],[2]])

print(np.dot(a,b))



# 파이썬으로 구현
import numpy as np

x = np.array([[1,2]])
w = np.array([[1,3,5],[2,4,6]])
b = np.array([[7,8,5]])

print(np.dot(x,w) + b)



# 가중의 합인 y값이 활성함수인 sigmoid 함수를 통과하면 어떤 값으로 출력되는지 z 값 구하기
import numpy as np

def sigmoid(x):

    return 1 / (1 + np.exp(-x))

x = np.array([[1,2]])
w = np.array([[1,3,5],[2,4,6]])
b = np.array([[7,8,5]])
y = np.dot(x,w) + b
z = sigmoid(y)

print(z)



#
import numpy as np

def sigmoid(x):

    return 1 / (1 + np.exp(-x))

x = np.array([[4.5,6.2]])
w1 = np.array([[0.1, 0.3],[0.2,0.4]])
b1 = np.array([0.7,0.8])
y1 = np.dot(x,w1) + b1
z1 = sigmoid(y1)

w2 = np.array([[0.5,0.6], [0.7,0.8]])
b2 = np.array([0.7,0.8])
y2 = np.dot(z1,w2) + b2
z2 = sigmoid(y2)

w3 = np.array([[0.1, 0.2], [0.3,0.4]])
b3 = np.array([0.7, 0.8])
y3 = np.dot(z2,w3) + b3

print(y3)


b = np.array([[7,8,5]])
y = np.dot(x,w) + b
z = sigmoid(y)



# 소프트 맥스 함수를 파이썬으로 구현
import numpy as np

def softmax(x):
    c = np.max(x)
    exp_a = np.exp(x-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

x = np.array([0.3, 2.9, 4.0])
y = softmax(x)

print(y)
print(sum(y))




# 항등함수를 파이썬으로 구현
def identify_function(x):

    return x



# 3층 신경망을 파이썬으로 구현
