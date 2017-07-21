# 평균 제곱 오차 함수
import numpy as np

def mean_squared_error(y,t):
    return 0.5 * np.sum((y-t) ** 2)

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]   # 숫자 2
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]

print(mean_squared_error(np.array(y1), np.array(t)))
print(mean_squared_error(np.array(y2), np.array(t)))



# 교차 엔트로피 오차 함수
import numpy as np

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y+delta))

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]   # 숫자 2
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]

print(cross_entropy_error(np.array(y1), np.array(t)))
print(cross_entropy_error(np.array(y2), np.array(t)))




# 오차율이 어떻게 되는지 loop 문을 사용해서 한번에 알아내기
import numpy as np

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y+delta))

t = [0,0,1,0,0,0,0,0,0,0]    # 숫자2

y1 = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
y2 = [0.1,0.05,0.2,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
y3 = [0.0,0.05,0.3,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
y4 = [0.0,0.05,0.4,0.0,0.05,0.0,0.0,0.5,0.0,0.0]
y5 = [0.0,0.05,0.5,0.0,0.05,0.0,0.0,0.4,0.0,0.0]
y6 = [0.0,0.05,0.6,0.0,0.05,0.0,0.0,0.3,0.0,0.0]
y7 = [0.0,0.05,0.7,0.0,0.05,0.0,0.0,0.2,0.0,0.0]
y8 = [0.0,0.1,0.8,0.0,0.1,0.0,0.0,0.2,0.0,0.0]
y9 = [0.0,0.05,0.9,0.0,0.05,0.0,0.0,0.0,0.0,0.0]



for i in range(1,10,1):
    print(cross_entropy_error(np.array(eval('y'+str(i))), np.array(t)))




# 60000 미만의 숫자중에 무작위로 10개 출력
import numpy as np

print(np.random.choice(60000, 10))




# mnist 60000장의 훈련 데이터중에 무작위로 10장을 골라내게하는 코드
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# print(x_train.shape)    # (60000, 784)
# print(t_train.shape)    # (60000, 10)

train_size = 60000
batch_size = 10

batch_mask = np.random.choice(train_size, batch_size)

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

print(len(x_batch))
print(x_batch.shape)



# 배치용 교차 엔트로피 오차 함수
import numpy as np

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y+delta)) / len(y)




# 교차 엔트로피 오차 사용 코드 구현!(mnist)
import sys,os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
import pickle

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
    return x_train,t_train

def softmax(a):
    a = a.T
    c = np.max(a, axis=0)
    a = a - c
    exp_a = np.exp(a)
    exp_sum = np.sum(exp_a, axis=0)
    y = exp_a/exp_sum
    return y.T

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)
    return y

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y+delta)) / len(y)

x, t = get_data()
network = init_network()

train_size = 60000
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)

x_batch = x[batch_mask]
t_batch = t[batch_mask]

y = predict(network, x_batch)
p = np.argmax(y, axis=1)

print(cross_entropy_error(y, t_batch))




# 근사로 구한 미분 함수 구현
import numpy as np

def numerical_diff(f, x):
    h = 1e-4

    return(f(x+h) - f(x-h)) / (2*h)




# y = 0.01x^2 + 0.1x 함수 미분하는데 x=10 일때 미분계수 구하기
import numpy as np

def numerical_diff(f, x):
    h = 1e-4

    return(f(x+h) - f(x-h)) / (2*h)

def function(x):

    return 0.01*x**2 + 0.1*x

print(numerical_diff(function, 10))






