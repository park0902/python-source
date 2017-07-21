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













import numpy as np

# a = np.array([[1,1,-1], [4,0,2], [1,0,0]])
# b = np.array([[2,-1],[3,-2],[0,1]])

a = np.matrix([[1,1,-1], [4,0,2], [1,0,0]])
b = np.matrix([[2,-1],[3,-2],[0,1]])

print(a*b)

print(np.dot(a,b))







import numpy as np

def softmax(x):
    c = np.max(x)
    exp_a = np.exp(x-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

x = np.array([[0.2,0.7, 0.9]])
w = np.array([[2,2,2],[4,3,4],[3,5,4]])
b = np.array([[-3,4,9]])
o = np.dot(x,w) + b
print(np.dot(x,w))
y = softmax(o)

print(y)




# 하나의 x[34]의 테스트 데이터가 신경망이 예측한것과 맞는지 확인
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import pickle               # mnist 데이터셋을 읽어올때 인터넷이 연결되어 있는 상태에서 가져와야하는데 이때 시간이 걸린다
                            # 가져온 이미지를 로컬에 저장할때 pickle 파일로 생성이 되고 로컬에 저장되어있으면 순식간에 읽을수 있다

from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


'''
MNIST를 위한 신경망 설계

 •입력층 : 784개 (이미지 크기가 28x28 픽셀임으로)
 •출력층 : 10개 (0에서 9로 분류함으로)
 •첫번째 은닉층 : 50개 (임의의 수)
 •두번째 은닉층 : 100개 (임의의 수)
'''


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()
accuracy_cnt = 0

y = predict(network, x[34])
p = np.argmax(y)
print('예측 :', p, ', 실제 :', t[34])



# 리스트를 만들고 이중에 최대값을 numpy 를 이용해 출력
import numpy as np

x = list(range(0,10,3))
y = max(x)

print(x.index(y))
print(np.argmax(list(range(0,10,3))))



# 행렬을 생성하고 각 행의 최대값에 해당하는 인덱스 출력
import numpy as np

x = np.array([[0.1, 0.8, 0.1], [0.3,0.1,0.6],[0.2,0.5,0.3], [0.8,0.1,0.1]])

for i in range(len(x)):
    print(np.argmax(x[i]))



# 2개의 리스트를 만들고 서로 같은 자리에 같은 숫자가 몇개 있는지 출력
import numpy as np

a = np.array([[2,1,3,5,1,4,2,1,1,0]])
b = np.array([[2,1,3,4,5,4,2,1,1,2]])

print(np.sum(a==b))



# 리스트를 x 라는 변수에 담고 앞에 5개의 숫자만 출력

x = [1,2,3,4,5,6,7,8,9,10]

print(x[:5])




# 100장의 이미지를 한번에 입력층에 넣어서 추론하는 신경망 코드 수행
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
import time
import psutil

# 시작 시간 체크
stime = time.time()
# 시작 메모리 체크 #################

proc1 = psutil.Process(os.getpid())
mem1 = proc1.memory_info()
before_start = mem1[0]





def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


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


x, t = get_data()
network = init_network()

batch_size = 100    # 배치 크기
accuracy_cnt = 0


# print(x[0])
# print(x[0:2])
# A = np.asanyarray(x[0]).reshape(28,28)
for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

# 종료 시간 체크
etime = time.time()
print('consumption time : ', round(etime-stime, 6))
# 실행 후 맨 밑에서 코드 구동 후 메모리 체크

proc = psutil.Process(os.getpid())
mem = proc.memory_info()
after_start = mem[0]
print('memory use : ', after_start-before_start)







# 훈련데이터(6만개)로 batch_size 를 1로 했을때와  100으로 했을때 수행 속도의 차이가 있는지 확인! (정확도와 수행속도)
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
import time
import psutil

# 시작 시간 체크
stime = time.time()
# 시작 메모리 체크 #################

proc1 = psutil.Process(os.getpid())
mem1 = proc1.memory_info()
before_start = mem1[0]





def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_train, t_train


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


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


x, t = get_data()
network = init_network()

batch_size = 100    # 배치 크기
accuracy_cnt = 0


# print(x[0])
# print(x[0:2])
# A = np.asanyarray(x[0]).reshape(28,28)
for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

# 종료 시간 체크
etime = time.time()
print('consumption time : ', round(etime-stime, 6))
# 실행 후 맨 밑에서 코드 구동 후 메모리 체크

proc = psutil.Process(os.getpid())
mem = proc.memory_info()
after_start = mem[0]
print('memory use : ', after_start-before_start)