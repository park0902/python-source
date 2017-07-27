# # 평균 제곱 오차 함수
# import numpy as np
#
# def mean_squared_error(y,t):
#     return 0.5 * np.sum((y-t) ** 2)
#
# t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]   # 숫자 2
# y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
# y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
#
# print(mean_squared_error(np.array(y1), np.array(t)))
# print(mean_squared_error(np.array(y2), np.array(t)))
#
#
#
# # 교차 엔트로피 오차 함수
# import numpy as np
#
# def cross_entropy_error(y, t):
#     delta = 1e-7
#     return -np.sum(t * np.log(y+delta))
#
# t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]   # 숫자 2
# y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
# y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
#
# print(cross_entropy_error(np.array(y1), np.array(t)))
# print(cross_entropy_error(np.array(y2), np.array(t)))
#
#
#
#
# # 오차율이 어떻게 되는지 loop 문을 사용해서 한번에 알아내기
# import numpy as np
#
# def cross_entropy_error(y, t):
#     delta = 1e-7
#     return -np.sum(t * np.log(y+delta))
#
# t = [0,0,1,0,0,0,0,0,0,0]    # 숫자2
#
# y1 = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
# y2 = [0.1,0.05,0.2,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
# y3 = [0.0,0.05,0.3,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
# y4 = [0.0,0.05,0.4,0.0,0.05,0.0,0.0,0.5,0.0,0.0]
# y5 = [0.0,0.05,0.5,0.0,0.05,0.0,0.0,0.4,0.0,0.0]
# y6 = [0.0,0.05,0.6,0.0,0.05,0.0,0.0,0.3,0.0,0.0]
# y7 = [0.0,0.05,0.7,0.0,0.05,0.0,0.0,0.2,0.0,0.0]
# y8 = [0.0,0.1,0.8,0.0,0.1,0.0,0.0,0.2,0.0,0.0]
# y9 = [0.0,0.05,0.9,0.0,0.05,0.0,0.0,0.0,0.0,0.0]
#
#
#
# for i in range(1,10,1):
#     print(cross_entropy_error(np.array(eval('y'+str(i))), np.array(t)))
#
#
#
#
# # 60000 미만의 숫자중에 무작위로 10개 출력
# import numpy as np
#
# print(np.random.choice(60000, 10))
#
#
#
#
# # mnist 60000장의 훈련 데이터중에 무작위로 10장을 골라내게하는 코드
# import sys, os
# sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록
# import numpy as np
# from dataset.mnist import load_mnist
#
# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
#
# # print(x_train.shape)    # (60000, 784)
# # print(t_train.shape)    # (60000, 10)
#
# train_size = 60000
# batch_size = 10
#
# batch_mask = np.random.choice(train_size, batch_size)
#
# x_batch = x_train[batch_mask]
# t_batch = t_train[batch_mask]
#
# print(len(x_batch))
# print(x_batch.shape)
#
#
#
# # 배치용 교차 엔트로피 오차 함수
# import numpy as np
#
# def cross_entropy_error(y, t):
#     delta = 1e-7
#     return -np.sum(t * np.log(y+delta)) / len(y)
#
#
#
#
# # 교차 엔트로피 오차 사용 코드 구현!(mnist)
# import sys,os
# sys.path.append(os.pardir)
# import numpy as np
# from dataset.mnist import load_mnist
# import pickle
#
# def get_data():
#     (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
#     return x_train,t_train
#
# def softmax(a):
#     a = a.T
#     c = np.max(a, axis=0)
#     a = a - c
#     exp_a = np.exp(a)
#     exp_sum = np.sum(exp_a, axis=0)
#     y = exp_a/exp_sum
#     return y.T
#
# def init_network():
#     with open("sample_weight.pkl", 'rb') as f:
#         network = pickle.load(f)
#     return network
#
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
#
# def predict(network, x):
#     w1, w2, w3 = network['W1'], network['W2'], network['W3']
#     b1, b2, b3 = network['b1'], network['b2'], network['b3']
#     a1 = np.dot(x, w1) + b1
#     z1 = sigmoid(a1)
#     a2 = np.dot(z1, w2) + b2
#     z2 = sigmoid(a2)
#     a3 = np.dot(z2, w3) + b3
#     y = softmax(a3)
#     return y
#
# def cross_entropy_error(y, t):
#     delta = 1e-7
#     return -np.sum(t * np.log(y+delta)) / len(y)
#
# x, t = get_data()
# network = init_network()
#
# train_size = 60000
# batch_size = 10
# batch_mask = np.random.choice(train_size, batch_size)
#
# x_batch = x[batch_mask]
# t_batch = t[batch_mask]
#
# y = predict(network, x_batch)
# p = np.argmax(y, axis=1)
#
# print(cross_entropy_error(y, t_batch))
#
#
#
#
# # 근사로 구한 미분 함수 구현
# import numpy as np
#
# def numerical_diff(f, x):
#     h = 1e-4
#
#     return(f(x+h) - f(x-h)) / (2*h)
#
#
#
#
# # y = 0.01x^2 + 0.1x 함수 미분하는데 x=10 일때 미분계수 구하기
# import numpy as np
#
# def numerical_diff(f, x):
#     h = 1e-4
#
#     return(f(x+h) - f(x-h)) / (2*h)
#
# def function(x):
#
#     return 0.01*x**2 + 0.1*x
#
# print(numerical_diff(function, 10))
#
#
#
#
#
#
#
# #
# import numpy as np
#
# def numerical_diff(f, x):
#     h = 1e-4
#
#     return(f(x+h) - f(x-h)) / (2*h)
#
# def function(x):
#
#     return 3*x**2 + 4*x
#
# print(numerical_diff(function, 7))
#
#
#
#
# #
# import numpy as np
#
# x = np.random.rand(100, 784)
#
# def softmax(a):
#     if a.ndim == 2:
#         a = a.T
#         c = np.max(a, axis=0)
#         a = a - c
#
#
#     return a.T
#
#
# x2 = softmax(x)
#
# print(sum(x[0]))
# print(sum(x2[0]))
#
#
#
# # y = x^2 + 4^2 의 함수를 미분해서 x=3 에서의 미분계수 구하기
# def numerical_diff(f, x):
#     h = 1e-4
#
#     return(f(x+h) - f(x-h)) / (2*h)
#
# def function(x):
#
#     return x**2 + 4**2
#
# print(numerical_diff(function, 4))
#
#
#
#
# # 문제75 아래의 함수 시각화
# import  numpy as np
# import matplotlib.pyplot as plt
#
# x = np.arange(0.0, 20.0, 0.1)
#
#
# def function(x):
#
#     return x**2 + 4**2
#
#
# y = function(x)
#
# plt.plot(x,y)
#
#
#
# # f(x0, x1) = x0^2 + x1^2  함수를 편미분하는데 x0=3, x1=4일때
# def numerical_diff(f, x):
#     h = 1e-4
#
#     return(f(x+h) - f(x-h)) / (2*h)
#
# def function_1(x0):
#
#     return x0*x0 + 4**2
#
# print(numerical_diff(function_1, 4))
#
#
#
#
# # 아래의 함수를 x0로 편미분(x0=3, x1=4)
# def numerical_diff(f, x):
#     h = 1e-4
#
#     return(f(x+h) - f(x-h)) / (2*h)
#
# def function_1(x0):
#
#     return 2*x0**2 + 12**2
#
# print(numerical_diff(function_1, 3))
#
#
#
#
# # 함수를 편미분!(x0=6, x1=7)
# def numerical_diff(f, x):
#     h = 1e-4
#
#     return(f(x+h) - f(x-h)) / (2*h)
#
# func = lambda x1:6*6*2 + 2*x1**2
#
# print(numerical_diff(func, 7))
#
#
#
#
# # for loop 문을 이용해서 함수를 x0, x1 로 편미분
# def numerical_diff(f, x):
#     h = 1e-4
#
#     return(f(x+h) - f(x-h)) / (2*h)
#
# func = (lambda x0:6*x0**2 + 2*7**2, lambda x1:6*6**2 + 2*x1**2)
# num= [6, 7]
#
# for f, n in zip(func,num):
#     print(numerical_diff(f, n))
#
#
#
#
# # 기울기 함수
# import numpy as np
# def numerical_gradient(f, x):   # x = np.array([30, 4.0])
#     h = 1e-4
#     grad = np.zeros_like(x)     # x 와 형상이 같은 배열 생성
#
#
#     for idx in range(x.size):   # x.size 에 2가 들어가니까 0, 1로 loop
#         tmp_val = x[idx]        # 3.0
#         x[idx] = tmp_val + h    # [3.0001, 4.0]
#         fxh1 = f(x)             # 25.0006
#         print(fxh1)
#         x[idx] = tmp_val - h    # [2.9999, 4.0]
#         fxh2 = f(x)             # 24.9994
#         print(fxh2)
#         grad[idx] = (fxh1 -fxh2) / (2*h)
#         x[idx] = tmp_val        # 원래 값으로 다시 복원원
#
#     return grad
#
# def func2(x):
#     return x[0]**2 + x[1]**2
#
# print(numerical_gradient(func2, np.array([3.0, 4.0])))
#
#
#
#
# # np.zeros_like 확인
# import numpy as np
#
# x = np.array([30, 4.0])
# grad = np.zeros_like(x)
#
# print(x.shape, grad.shape)
#
#
#
#
# # x0=0.0, x1=2.0 일때의 기울기 벡터 구하기
# import numpy as np
# def numerical_gradient(f, x):   # x = np.array([30, 4.0])
#     h = 1e-4
#     grad = np.zeros_like(x)     # x 와 형상이 같은 배열 생성
#
#
#     for idx in range(x.size):   # x.size 에 2가 들어가니까 0, 1로 loop
#         tmp_val = x[idx]
#         x[idx] = tmp_val + h
#         fxh1 = f(x)
#         x[idx] = tmp_val - h
#         fxh2 = f(x)
#         grad[idx] = (fxh1 -fxh2) / (2*h)
#         x[idx] = tmp_val        # 원래 값으로 다시 복원원
#
#     return grad
#
# def func2(x):
#     return x[0]**2 + x[1]**2
#
# print(numerical_gradient(func2, np.array([0.0, 2.0])))
#
#
#
#
# # 경사 감소 함수 파이썬으로 구현
# import numpy as np
#
# init_x = np.array([-3.0, 4.0])
#
#
# def func2(x):
#     return x[0]**2 + x[1]**2
#
#
# def numerical_gradient(f, x):
#     h = 1e-4
#     grad = np.zeros_like(x)     # x 와 형상이 같은 배열 생성
#
#
#     for idx in range(x.size):
#         tmp_val = x[idx]
#         x[idx] = tmp_val + h
#         fxh1 = f(x)
#         x[idx] = tmp_val - h
#         fxh2 = f(x)
#         grad[idx] = (fxh1 -fxh2) / (2*h)
#         x[idx] = tmp_val        # 원래 값으로 다시 복원
#
#     return grad
#
# def gradient_descent(f, init_x, lr=0.01, step_num=100):
#     x = init_x
#
#     for i in range(step_num):
#         grad = numerical_gradient(f, x)
#         x -= lr * grad
#
#     return x
#
# print(gradient_descent(func2, init_x, lr=0.1, step_num=100))
#
#
#
#
# # 학습률이 너무 크면 (10) 이면 발산을 하고 학습률이 너무 작으면 (1e-10) 으로 수렴을 못한다는 것을 테스트
# import numpy as np
#
# def func2(x):
#     return x[0]**2 + x[1]**2
#
#
# def numerical_gradient(f, x):
#     h = 1e-4
#     grad = np.zeros_like(x)     # x 와 형상이 같은 배열 생성
#
#
#     for idx in range(x.size):
#         tmp_val = x[idx]
#         x[idx] = tmp_val + h
#         fxh1 = f(x)
#         x[idx] = tmp_val - h
#         fxh2 = f(x)
#         grad[idx] = (fxh1 -fxh2) / (2*h)
#         x[idx] = tmp_val        # 원래 값으로 다시 복원
#
#     return grad
#
# def gradient_descent(f, init_x, lr=0.01, step_num=100):
#     x = init_x
#
#     for i in range(step_num):
#         grad = numerical_gradient(f, x)
#         x -= lr * grad
#
#     return x
#
# init_x = np.array([-3.0, 4.0])
# print('학습률 10 일때 : ', gradient_descent(func2, init_x, lr=10.0, step_num=100))
#
# init_x = np.array([-3.0, 4.0])
# print('학습률 1e-10 일때 : ', gradient_descent(func2, init_x, lr=1e-10, step_num=100))
#
#
#
#
# # 2X3 의 가중치를 랜덤으로 생성하고 간단한 신경망을 구현
# import sys,os
# sys.path.append(os.pardir)
# import numpy as np
# from common.functions import softmax, cross_entropy_error
# from common.gradient import numerical_gradient
#
# class simpleNet:
#     def __init__(self):
#         self.W = np.random.rand(2,3)     # 2x3의 가중치 배열을 랜덤 생성
#
#     def predit(self, x):
#         return np.dot(x, self.W)
#
#     def loss(self,x,t):
#         z = self.predit(x)
#         y = softmax(z)
#         print(y)
#         loss = cross_entropy_error(y,t)
#
#         return loss
#
# net = simpleNet()
#
# x = np.array([0.6, 0.9])
# p = net.predit(x)
# print(p)
# t = np.array([0,0,1])
# print(net.loss(x,t))
#
#
#
# # 수치미분함수에 신경망의 비용함수와 가중치(2x3) 의 가중치를 입력해서 기울기(2x3) 구하기
# import sys,os
# sys.path.append(os.pardir)
# import numpy as np
# from common.functions import softmax, cross_entropy_error
# from common.gradient import numerical_gradient
#
#
# class simpleNet:
#     def __init__(self):
#         self.W = np.random.rand(2,3)     # 2x3의 가중치 배열을 랜덤 생성
#
#     def predit(self, x):
#         return np.dot(x, self.W)
#
#     def loss(self,x,t):
#         z = self.predit(x)
#         y = softmax(z)
#         loss = cross_entropy_error(y,t)
#
#         return loss
#
# net = simpleNet()
#
# x = np.array([0.6, 0.9])
# p = net.predit(x)
# t = np.array([0,0,1])
#
# # def f(W):
# #     return net.loss(x,t)
#
#
# f = lambda W:net.loss(x,t)
#
# dW = numerical_gradient(f, net.W)
#
# print(dW)
#
#
#
#
#
# # 2층 신경망
# import sys, os
# sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
# import numpy as np
# import matplotlib.pyplot as plt
# from dataset.mnist import load_mnist
# from common.functions import *
# from common.gradient import numerical_gradient
#
#
# class TwoLayerNet:
#     def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
#         self.params = {}
#         self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
#         self.params['b1'] = np.zeros(hidden_size)
#         self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
#         self.params['b2'] = np.zeros(output_size)
#
#     def predict(self, x):
#         W1, W2 = self.params['W1'], self.params['W2']
#         b1, b2 = self.params['b1'], self.params['b2']
#         a1 = np.dot(x, W1) + b1
#         z1 = sigmoid(a1)
#         a2 = np.dot(z1, W2) + b2
#         y = softmax(a2)
#         return y
#
#     def loss(self, x,t):
#         y = self.predict(x)
#         return cross_entropy_error(y, t)
#
#     def accuracy(self, x, t):
#         y = self.predict(x)
#         y = np.argmax(y, axis=1)
#         t = np.argmax(t, axis=1)
#         accuracy = np.sum(y == t) / float(x.shape[0])
#         return accuracy
#
#     def numerical_gradient(self,x,t):
#         loss_W = lambda W: self.loss(x,t)
#         grads = {}
#         grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
#         grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
#         grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
#         grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
#         return grads
#
#
#
#
#
# x(입력값), t(target 값), y(예상값)을 2층 신경망을 객체화해서 W1, W2, b1, b2 의 차원이 어떻게 되는지 출력
# import sys, os
# sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
# import numpy as np
# from dataset.mnist import load_mnist
# from common.functions import *
# from common.gradient import numerical_gradient
#
#
# class TwoLayerNet:
#     def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
#         self.params = {}
#         self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
#         self.params['b1'] = np.zeros(hidden_size)
#         self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
#         self.params['b2'] = np.zeros(output_size)
#
#     def predict(self, x):
#         W1, W2 = self.params['W1'], self.params['W2']
#         b1, b2 = self.params['b1'], self.params['b2']
#         a1 = np.dot(x, W1) + b1
#         z1 = sigmoid(a1)
#         a2 = np.dot(z1, W2) + b2
#         y = softmax(a2)
#         return y
#
#     def loss(self, x,t):
#         y = self.predict(x)
#         return cross_entropy_error(y, t)
#
#     def accuracy(self, x, t):
#         y = self.predict(x)
#         y = np.argmax(y, axis=1)
#         t = np.argmax(t, axis=1)
#         accuracy = np.sum(y == t) / float(x.shape[0])
#         return accuracy
#
#     def numerical_gradient(self,x,t):
#         loss_W = lambda W: self.loss(x,t)
#         grads = {}
#         grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
#         grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
#         grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
#         grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
#         return grads
#
# net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
#
# x = np.random.rand(100,784)
# y = net.predict(x)
# t = np.random.rand(100,10)
#
# grads = net.numerical_gradient(x,t)
#
# print(grads['W1'].shape)
# print(grads['W2'].shape)
# print(grads['b1'].shape)
# print(grads['b2'].shape)





#
# import sys, os
# sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
# import numpy as np
# import matplotlib.pyplot as plt
# from dataset.mnist import load_mnist
# from common.functions import *
# from common.gradient import numerical_gradient
#
#
# class TwoLayerNet:
#     def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
#         self.params = {}
#         self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
#         self.params['b1'] = np.zeros(hidden_size)
#         self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
#         self.params['b2'] = np.zeros(output_size)
#
#     def predict(self, x):
#         W1, W2 = self.params['W1'], self.params['W2']
#         b1, b2 = self.params['b1'], self.params['b2']
#         a1 = np.dot(x, W1) + b1
#         z1 = sigmoid(a1)
#         a2 = np.dot(z1, W2) + b2
#         y = softmax(a2)
#         return y
#
#     def loss(self, x,t):
#         y = self.predict(x)
#         return cross_entropy_error(y, t)
#
#     def accuracy(self, x, t):
#         y = self.predict(x)
#         y = np.argmax(y, axis=1)
#         t = np.argmax(t, axis=1)
#         accuracy = np.sum(y == t) / float(x.shape[0])
#         return accuracy
#
#     def numerical_gradient(self,x,t):
#         loss_W = lambda W: self.loss(x,t)
#         grads = {}
#         grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
#         grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
#         grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
#         grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
#         return grads
#
# # 데이터 읽기
# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
#
#
# # 훈련과 테스트 데이터를 같이 한번에 정확도를 계산하려고 빈 리스트를 만들고 있는데
# # 왜 두개를 같이 확인하나면 훈련 데이터가 혹시 오퍼피팅 되지는 않았는지 확인하려고
#
#
# # 하이퍼파라미터
# iters_num = 10000  # 반복 횟수를 적절히 설정한다.
# train_size = x_train.shape[0] # 60000 개
# # print(x_train.shape[1])  # 784개
# batch_size = 100  # 미니배치 크기
# learning_rate = 0.1
# train_loss_list = []
# train_acc_list = []
# test_acc_list = []
#
# # 1에폭당 반복 수
# iter_per_epoch = max(train_size / batch_size, 1)
# # print(iter_per_epoch) # 600
#
# for i in range(iters_num): # 10000
#     # 미니배치 획득  # 랜덤으로 100개씩 뽑아서 10000번을 수행하니까 백만번
#     batch_mask = np.random.choice(train_size, batch_size) # 100개 씩 뽑아서 10000번 백만번
#     x_batch = x_train[batch_mask]
#     # print(x_batch.shape) #100 x 784
#     t_batch = t_train[batch_mask]
#     # print(t_batch.shape) # 100 x 10
#
#     # 기울기 계산
#     grad = network.numerical_gradient(x_batch, t_batch)
#     #grad = network.gradient(x_batch, t_batch)
#     # 매개변수 갱신
#
#     for key in ('W1', 'b1', 'W2', 'b2'):
#         network.params[key] -= learning_rate * grad[key]
#
#     # 학습 경과 기록
#     loss = network.loss(x_batch, t_batch)
#     train_loss_list.append(loss) # cost 가 점점 줄어드는것을 보려고
#     # 1에폭당 정확도 계산 # 여기는 훈련이 아니라 1에폭 되었을때 정확도만 체크
#
#     if i % iter_per_epoch == 0: # 600 번마다 정확도 쌓는다.
#         # print(x_train.shape) # 60000,784
#         train_acc = network.accuracy(x_train, t_train)
#         test_acc = network.accuracy(x_test, t_test)
#         train_acc_list.append(train_acc) # 10000/600 개  16개 # 정확도가 점점 올라감
#         test_acc_list.append(test_acc)  # 10000/600 개 16개 # 정확도가 점점 올라감
#         print(i)
#         print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
#
# # 그래프 그리기
# markers = {'train': 'o', 'test': 's'}
# x = np.arange(len(train_acc_list))
# plt.plot(x, train_acc_list, label='train acc')
# plt.plot(x, test_acc_list, label='test acc', linestyle='--')
# plt.xlabel("epochs")
# plt.ylabel("accuracy")
# plt.ylim(0, 1.0)
# plt.legend(loc='lower right')
# plt.show()







import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.functions import *
from common.gradient import numerical_gradient


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y

    def loss(self, x,t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)


# 훈련과 테스트 데이터를 같이 한번에 정확도를 계산하려고 빈 리스트를 만들고 있는데
# 왜 두개를 같이 확인하나면 훈련 데이터가 혹시 오퍼피팅 되지는 않았는지 확인하려고


# 하이퍼파라미터
iters_num = 10000  # 반복 횟수를 적절히 설정한다.
train_size = x_train.shape[0] # 60000 개
# print(x_train.shape[1])  # 784개
batch_size = 100  # 미니배치 크기
learning_rate = 0.1
train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)
# print(iter_per_epoch) # 600

for i in range(iters_num): # 10000
    # 미니배치 획득  # 랜덤으로 100개씩 뽑아서 10000번을 수행하니까 백만번
    batch_mask = np.random.choice(train_size, batch_size) # 100개 씩 뽑아서 10000번 백만번
    x_batch = x_train[batch_mask]
    # print(x_batch.shape) #100 x 784
    t_batch = t_train[batch_mask]
    # print(t_batch.shape) # 100 x 10

    # 기울기 계산
    grad = network.gradient(x_batch, t_batch)
    #grad = network.gradient(x_batch, t_batch)
    # 매개변수 갱신

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss) # cost 가 점점 줄어드는것을 보려고
    # 1에폭당 정확도 계산 # 여기는 훈련이 아니라 1에폭 되었을때 정확도만 체크

    if i % iter_per_epoch == 0: # 600 번마다 정확도 쌓는다.
        # print(x_train.shape) # 60000,784
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc) # 10000/600 개  16개 # 정확도가 점점 올라감
        test_acc_list.append(test_acc)  # 10000/600 개 16개 # 정확도가 점점 올라감

        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()






import tensorflow as tf

tf.train.ExponentialMovingAverage
