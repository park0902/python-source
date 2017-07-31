# # # 곱셈계층 class 구현
# # class MulLayer:
# #     def __init__(self):
# #         self.x = None
# #         self.y = None
# #
# #     def forward(self, x, y):
# #         self.x = x
# #         self.y = y
# #         out = x * y
# #
# #         return out
# #
# #     def backward(self, dout):
# #         dx = dout * self.y
# #         dy = dout * self.x
# #
# #         return dx, dy
# #
# #
# #
# #
# # # 곱셈 클래스를 객체화 시켜서 사과가격 구하기(순전파)
# # class MulLayer:
# #     def __init__(self):
# #         self.x = None
# #         self.y = None
# #
# #     def forward(self, x, y):
# #         self.x = x
# #         self.y = y
# #         out = x * y
# #
# #         return out
# #
# #     def backward(self, dout):
# #         dx = dout * self.y
# #         dy = dout * self.x
# #
# #         return dx, dy
# #
# # apple = 200
# # apple_num = 5
# # tax = 1.2
# #
# # mul_apple_layer = MulLayer()
# # mul_apple_tax_layer = MulLayer()
# #
# # apple_price = mul_apple_layer.forward(apple, apple_num)
# # price = mul_apple_tax_layer.forward(apple_price, tax)
# #
# # print(price)
# #
# #
# #
# #
# # # 덧셈계층 class 구현
# # class AddLayer:
# #     def __init__(self):
# #         pass
# #
# #     def foward(self, x, y):
# #         out = x + y
# #         return out
# #
# #     def backward(self,dout):
# #         dx = dout * 1
# #         dy = dout * 1
# #
# #         return dx, dy
# #
# #
# #
# #
# # # 사과 2개와 귤 5개를 구입하면 총 가격이 얼마인지 구하시오
# # class MulLayer:
# #     def __init__(self):
# #         self.x = None
# #         self.y = None
# #
# #     def forward(self, x, y):
# #         self.x = x
# #         self.y = y
# #         out = x * y
# #
# #         return out
# #
# #     def backward(self, dout):
# #         dx = dout * self.y
# #         dy = dout * self.x
# #
# #         return dx, dy
# #
# #
# # class AddLayer:
# #     def __init__(self):
# #         pass
# #
# #     def foward(self, x, y):
# #         out = x + y
# #         return out
# #
# #     def backward(self, dout):
# #         dx = dout
# #         dy = dout
# #
# #         return dx, dy
# #
# #
# #
# #
# # # 사과 2개와 귤 5개를 구입하면 총 가격이 얼마인지 구하시오
# # class MulLayer:
# #     def __init__(self):
# #         self.x = None
# #         self.y = None
# #
# #     def forward(self, x, y):
# #         self.x = x
# #         self.y = y
# #         out = x * y
# #
# #         return out
# #
# #     def backward(self, dout):
# #         dx = dout * self.y
# #         dy = dout * self.x
# #
# #         return dx, dy
# #
# #
# # class AddLayer:
# #     def __init__(self):
# #         pass
# #
# #     def foward(self, x, y):
# #         out = x + y
# #         return out
# #
# #     def backward(self, dout):
# #         dx = dout
# #         dy = dout
# #
# #         return dx, dy
# #
# # apple = 200
# # apple_num = 2
# # orange = 300
# # orange_num = 5
# # tax = 1.5
# #
# # mul_apple_layer = MulLayer()
# # mul_orange_layer = MulLayer()
# # mul_tax_layer = MulLayer()
# # add_apple_orange_layer = AddLayer()
# #
# # apple_price = mul_apple_layer.forward(apple, apple_num)
# # orange_price = mul_orange_layer.forward(orange, orange_num)
# # apple_orange_price = add_apple_orange_layer.foward(apple_price, orange_price)
# # all_price = mul_tax_layer.forward(apple_orange_price, tax)
# #
# # print(all_price)
# #
#
#
#
#
# # 역전파 구현
# class MulLayer:
#     def __init__(self):
#         self.x = None
#         self.y = None
#
#     def forward(self, x, y):
#         self.x = x
#         self.y = y
#         out = x * y
#
#         return out
#
#     def backward(self, dout):
#         dx = dout * self.y
#         dy = dout * self.x
#
#         return dx, dy
#
#
# class AddLayer:
#     def __init__(self):
#         pass
#
#     def forward(self, x, y):
#         out = x + y
#         return out
#
#     def backward(self, dout):
#         dx = dout
#         dy = dout
#
#         return dx, dy
#
# apple = 200
# apple_num = 2
# orange = 300
# orange_num = 5
# tax = 1.5
# dprice = 1
#
# mul_apple_layer = MulLayer()
# mul_orange_layer = MulLayer()
# add_apple_orange_layer = AddLayer()
# mul_tax_layer = MulLayer()
#
#
# apple_price = mul_apple_layer.forward(apple, apple_num)
# orange_price = mul_orange_layer.forward(orange, orange_num)
# apple_orange_price = add_apple_orange_layer.forward(apple_price, orange_price)
# all_price = mul_tax_layer.forward(apple_orange_price, tax)
#
#
#
# dall_price, dtax = mul_tax_layer.backward(1)
# dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
# dorange, dorange_num = mul_orange_layer.backward(dorange_price)
# dapple, dapple_num = mul_apple_layer.backward(dapple_price)
#
# print(dapple_num, dapple, dorange, dorange_num, dtax)
#
#
#
#
# # 순전파 구현
# class MulLayer:
#     def __init__(self):
#         self.x = None
#         self.y = None
#
#     def forward(self, x, y):
#         self.x = x
#         self.y = y
#         out = x * y
#
#         return out
#
#     def backward(self, dout):
#         dx = dout * self.y
#         dy = dout * self.x
#
#         return dx, dy
#
#
# class AddLayer:
#     def __init__(self):
#         pass
#
#     def forward(self, x, y):
#         out = x + y
#         return out
#
#     def backward(self, dout):
#         dx = dout
#         dy = dout
#
#         return dx, dy
#
# apple, apple_num = 100, 4
# orange, orange_num = 200, 3
# pear, pear_num = 300, 2
# tax = 1.3
#
# mul_apple_layer = MulLayer()
# mul_orange_layer = MulLayer()
# mul_pear_layer = MulLayer()
# add_apple_orange_peer_layer = AddLayer()
# mul_tax_layer = MulLayer()
#
# a = mul_apple_layer.forward(apple, apple_num)
# b = mul_orange_layer.forward(orange, orange_num)
# c = mul_pear_layer.forward(pear, pear_num)
#
# d = add_apple_orange_peer_layer.forward(a,b)
# e = add_apple_orange_peer_layer.forward(c,d)
#
# all_price = mul_tax_layer.forward(e, tax)
#
# print(all_price)
#
#
#
# # 역전파 구현
# class MulLayer:
#     def __init__(self):
#         self.x = None
#         self.y = None
#
#     def forward(self, x, y):
#         self.x = x
#         self.y = y
#         out = x * y
#
#         return out
#
#     def backward(self, dout):
#         dx = dout * self.y
#         dy = dout * self.x
#
#         return dx, dy
#
#
# class AddLayer:
#     def __init__(self):
#         pass
#
#     def forward(self, x, y):
#         out = x + y
#         return out
#
#     def backward(self, dout):
#         dx = dout
#         dy = dout
#
#         return dx, dy
#
# apple, apple_num = 100, 4
# orange, orange_num = 200, 3
# pear, pear_num = 300, 2
# tax = 1.3
#
# mul_apple_layer = MulLayer()
# mul_orange_layer = MulLayer()
# mul_pear_layer = MulLayer()
# add_all_layer = AddLayer()
# mul_tax_layer = MulLayer()
#
# a = mul_apple_layer.forward(apple, apple_num)
# b = mul_orange_layer.forward(orange, orange_num)
# c = mul_pear_layer.forward(pear, pear_num)
#
# d = add_all_layer.forward(a,b)
# e = add_all_layer.forward(c,d)
#
# all_price = mul_tax_layer.forward(e, tax)
#
# dprice = 1
# dall_price, dtax = mul_tax_layer.backward(dprice)
#
#
#
# # numpy에서의 copy
# import numpy as np
# import copy
#
# x = np.array([[1.0, -0.5], [-2.0, 3.0]])
# print(x)
#
# mask = (x <= 0)
# print(mask)
#
# out = x.copy()
# print(out)
#
# out[mask] = 0   # 0 이하인것은 다 0 으로 변경해주는 작업
# print(out)
#
# print(x)        # copy 가 되었기 때문에 별도의 객체인 out 이 생성
#                 # x 객체와는 별도로 out 을 따로 변경한 것이다
#
#
# # ReLu class
# class Relu:
#     def __init__(self):
#         self.mask = None
#
#     def forward(self, x):
#         self.mask = (x <= 0)
#         out = x.copy()
#         out[self.mask] = 0
#
#         return out
#
#     def backward(self, dout):
#         dout[self.mask] = 0
#         dx = dout
#
#         return dx
#
#
#
#
# # x 변수를 생성하고 x 를 ReLu 객체에 forward 함수에 값을 넣으면 무엇이 출력되는지 확인
# class Relu:
#     def __init__(self):
#         self.mask = None
#
#     def forward(self, x):
#         self.mask = (x <= 0)
#         out = x.copy()
#         out[self.mask] = 0
#
#         return out
#
#     def backward(self, dout):
#         dout[self.mask] = 0
#         dx = dout
#
#         return dx
#
# relu = Relu()
#
# x = np.array([[1.0, -0.5], [-2.0, 3.0]])
#
# print(relu.forward(x))
#
#
#
# # Sigmoid class
# import numpy as np
#
# class sigmoid:
#     def __init__(self):
#         self.out = None
#
#     def forward(self, x):
#         out = 1 / (1 + np.exp(-x))
#         self.out = out
#
#         return out
#
#     def backward(self, dout):
#         dx = dout * (1.0 - self.out) * self.out
#
#         return dx
#
#
#
#
#
# #
# import numpy as np
#
# x = np.array([5,6])
# w = np.array([[2,4,4,],[6,3,5]])
#
# print(np.dot(x,w))
#
#
#
#
# # numpy 를 이용해 행렬 내적 구하기
# import numpy as np
#
# x = np.array([[1,2,3], [4,5,6]])
# y = np.array([[1,2], [3,4], [5,6]])
#
# print(np.dot(x,y))
#
#
# import numpy as np
#
# x = np.array([[1,2], [3,4], [5,6]])
# y = np.array([7,8])
# y2 = np.array([7,8], ndmin=2)
# w = np.dot(x,y)
#
# print(x.shape)
# print(y.shape)
# print(y2.shape)
# print(w.shape)
# print(w)
#
#
# import numpy as np
#
# x = np.array([[1,2],[3,4],[5,6]])
# w = np.array([[1,2,3,4],[5,6,7,8]])
#
# print(np.dot(x,w))
#
#
#
#
# # 신경망을 행렬의 내적으로 구현해서 출력값 y 출력
# import numpy as np
#
# x = np.array([1,2])
# w = np.array([[1,3,5], [2,4,6]])
#
# print(np.dot(x,w))
#
#
#
#
# # 순전파를 구하는 함수 forward 이름으로 생성
# import numpy as np
#
# def forward(x,w,b):
#     out = np.dot(x,w) + b
#     return out
#
# x = np.array([1,2])
# w = np.array([[1,3,5], [2,4,6]])
# b = np.array([1,2,3])
#
# print(forward(x,w,b))
#
#
#
# # 역전파를 구하는 함수 backward 이름으로 생성
# import numpy as np
#
# def backward(x,w,out):
#     dx = np.dot(out, w.T)
#     dw = np.dot(x.T, out)
#     db = np.sum(out, axis=0)
#
#     return dx, dw, db
#
# x = np.array([1,2], ndmin=2)
# w = np.array([[1,3,5], [2,4,6]])
# out = np.array([6,13,20], ndmin=2)
# b = np.array([1,2,3])
#
# print(backward(x,w,out))
#
#
#
# # Affine class 생성
# import numpy as np
#
# class Affine:
#     def __init__(self,w,b):
#         self.w = w
#         self.b = b
#
#
#     def forward(self,x):
#         return np.dot(x, self.w) + self.b
#
#
#
#     def backward(self, x, out):
#         self.dx = np.dot(out, self.w.T)
#         self.dw = np.dot(x.T, out)
#         self.db = np.sum(out, axis=0)
#
#         return self.dx, self.dw, self.db
#
# x = np.array([1,2], ndmin=2)
# b = np.array([1,2,3])
# w = np.array([[1,3,5], [2,4,6]])
#
# affine1 = Affine(w,b)
#
# print(affine1.forward(x))
# print(affine1.backward(x,out))
#
#
#
# # 2층 신경망의 순전파를 Affine 클래스를 사용해서 출력
# import numpy as np
#
# class Affine:
#     def __init__(self,w,b):
#         self.w = w
#         self.b = b
#
#
#     def forward(self,x):
#         return np.dot(x, self.w) + self.b
#
#
#
#     def backward(self, x, out):
#         self.dx = np.dot(out, self.w.T)
#         self.dw = np.dot(x.T, out)
#         self.db = np.sum(out, axis=0)
#
#         return self.dx, self.dw, self.db
#
# x = np.array([1,2])
# w1 = np.array([[1,3,5],[2,4,6]])
# w2 = np.array([[1,4],[2,5],[3,6]])
# b1 = np.array([1,2,3])
# b2 = np.array([1,2])
#
# affine1 = Affine(w1,b1)
# affine12 = Affine(w2,b2)
# out = affine1.forward(x)
#
# print(affine12.forward(out))
#
#
#
#
# # 2층 신경망의 역전파를 Affine 클래스를 사용해서 출력
# import numpy as np
#
# class Affine:
#     def __init__(self,w,b):
#         self.w = w
#         self.b = b
#
#
#     def forward(self,x):
#         return np.dot(x, self.w) + self.b
#
#
#
#     def backward(self, x, out):
#         self.dx = np.dot(out, self.w.T)
#         self.dw = np.dot(x.T, out)
#         self.db = np.sum(out, axis=0)
#
#         return self.dx, self.dw, self.db
#
# x = np.array([1,2], ndmin=2)
# w1 = np.array([[1,3,5],[2,4,6]])
# w2 = np.array([[1,4],[2,5],[3,6]])
# b1 = np.array([1,2,3])
# b2 = np.array([1,2])
#
# affine1 = Affine(w1,b1)
# affine2 = Affine(w2,b2)
# out1 = affine1.forward(x)
# out2 = affine2.forward(out1)
#
# dx2, dw2, db2 = affine2.backward(out1, out2)
# dx1, dw1, db1 = affine1.backward(x, dx2)
#
# print(dx1, dw1, db1)
#
#
#
#
# # 2층 신경망의 순전파를 구현하는데 은닉층 활성화 함수로 ReLu 함수 추가해서 구현
# class Relu:
#     def __init__(self):
#         self.mask = None
#
#     def forward(self, x):
#         self.mask = (x <= 0)
#         out = x.copy()
#         out[self.mask] = 0
#
#         return out
#
#     def backward(self, dout):
#         dout[self.mask] = 0
#         dx = dout
#
#         return dx
#
#
# class Affine:
#     def __init__(self,w,b):
#         self.w = w
#         self.b = b
#
#
#     def forward(self,x):
#         return np.dot(x, self.w) + self.b
#
#
#
#     def backward(self, x, out):
#         self.dx = np.dot(out, self.w.T)
#         self.dw = np.dot(x.T, out)
#         self.db = np.sum(out, axis=0)
#
#         return self.dx, self.dw, self.db
#
# x = np.array([1,2], ndmin=2)
# w1 = np.array([[1,3,5],[2,4,6]])
# w2 = np.array([[1,4],[2,5],[3,6]])
# b1 = np.array([1,2,3])
# b2 = np.array([1,2])
#
# relu = Relu()
# affine1 = Affine(w1,b1)
# affine2 = Affine(w2,b2)
#
# out1 = affine1.forward(x)
# a_r = relu.forward(out1)
# out2 = affine2.forward(a_r)
#
# print(out2)
#
#
#
#
# # 2층 신경망의 역전파를 구현하는데 은닉층 활성화 함수로 ReLu 함수 추가해서 구현
# import numpy as np
#
# class Relu:
#     def __init__(self):
#         self.mask = None
#
#     def forward(self, x):
#         self.mask = (x <= 0)
#         out = x.copy()
#         out[self.mask] = 0
#
#         return out
#
#     def backward(self, dout):
#         dout[self.mask] = 0
#         dx = dout
#
#         return dx
#
#
# class Affine:
#     def __init__(self,w,b):
#         self.w = w
#         self.b = b
#
#
#     def forward(self,x):
#         return np.dot(x, self.w) + self.b
#
#
#
#     def backward(self, x, out):
#         self.dx = np.dot(out, self.w.T)
#         self.dw = np.dot(x.T, out)
#         self.db = np.sum(out, axis=0)
#
#         return self.dx, self.dw, self.db
#
#
# x = np.array([1,2], ndmin=2)
# w1 = np.array([[1,3,5],[2,4,6]])
# w2 = np.array([[1,4],[2,5],[3,6]])
# b1 = np.array([1,2,3])
# b2 = np.array([1,2])
#
# affine1 = Affine(w1, b1)
# affine2 = Affine(w2, b2)
# relu1 = Relu()
#
# out = affine1.forward(x)
# out_act = relu1.forward(out)
# out2 = affine2.forward(out_act)
# # print(out2)
#
# dx2, dw2, db2 = affine2.backward(out_act, out2)
# dx1 = relu1.backward(dx2)
# dx, dw, db = affine1.backward(x, dx1)
# print('dx\n', dx, '\ndw\n', dw, '\ndb\n', db)
#
#
#
#
#
# #
# import numpy as np
#
# class Affine:
#     def __init__(self,w,b):
#         self.w = w
#         self.b = b
#
#
#     def forward(self,x):
#         return np.dot(x, self.w) + self.b
#
#
#
#     def backward(self, x, out):
#         self.dx = np.dot(out, self.w.T)
#         self.dw = np.dot(x.T, out)
#         self.db = np.sum(out, axis=0)
#
#         return self.dx, self.dw, self.db
#
# x = np.array([[1,2], [2,4]])
# w = np.array([[1,3,5], [2,4,6]])
# b = np.array([1,2,3])
# out = affine1.forward(x)
#
# affine1 = Affine(w,b)
#
# # print(affine1.forward(x))
# print(affine1.backward(x,out))
#
#
#
#
# # softmaxwithloss 클래스
# def cross_entropy_error(y, t):
#     delta = 1e-7
#     return -np.sum(t * np.log(y+delta))
#
# def softmax(x):
#     c = np.max(x)
#     exp_a = np.exp(x-c)
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a / sum_exp_a
#
#     return y
#
#
# class SoftmaxWithLoss:
#     def __init__(self):
#         self.loss = None
#         self.y = None
#         self.t = None
#
#     def forward(self, x, t):
#         self.t = t
#         self.y = softmax()
#         self.loss = cross_entropy_error(self.y, self.t)
#         return self.loss
#
#     def backward(self, dout=1):
#         batch_size = self.t.shape[0]
#         dx = (self.y - self.t) / batch_size
#         return dx
#
#
#
# # x(입력값), t(targer value) 를 입력해서 순전파 오차율 확인
# import numpy as np
#
# def cross_entropy_error(y, t):
#     delta = 1e-7
#     return -np.sum(t * np.log(y+delta))
#
# def softmax(x):
#     c = np.max(x)
#     exp_a = np.exp(x-c)
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a / sum_exp_a
#
#     return y
#
#
# class SoftmaxWithLoss:
#     def __init__(self):
#         self.loss = None
#         self.y = None
#         self.t = None
#
#     def forward(self, x, t):
#         self.t = t
#         self.y = softmax(x)
#         self.loss = cross_entropy_error(self.y, self.t)
#         return self.loss
#
#     def backward(self, dout=1):
#         batch_size = self.t.shape[0]
#         dx = (self.y - self.t) / batch_size
#         return dx
#
# t = np.array([0,0,1,0,0,0,0,0,0,0])
# x = np.array([0.01,0.01,0.01,0.01,0.01,0.01,0.05,0.3,0.1,0.5])
# x2 = np.array([0.01,0.01,0.9,0.01,0.01,0.01,0.01,0.01,0.01,0.02])
# swl = SoftmaxWithLoss()
#
# print(swl.forward(x,t))
# print(swl.backward())
#
# print(swl.forward(x2,t))
# print(swl.backward())
#
#
#
# import collections
#
# print('dict : ')
#
# d1 = {}
# d1['a'] = 'A'
# d1['b'] = 'B'
# d1['c'] = 'C'
# d1['d'] = 'D'
# d1['e'] = 'E'
#
# d2 = {}
# d2['e'] = 'E'
# d2['d'] = 'D'
# d2['c'] = 'C'
# d2['b'] = 'B'
# d2['a'] = 'A'
#
# print(d1==d2)
#
#
# print('\nOrderdDict : ')
#
# d1 = collections.OrderedDict()
# d1['a'] = 'A'
# d1['b'] = 'B'
# d1['c'] = 'C'
# d1['d'] = 'D'
# d1['e'] = 'E'
#
# d2 = collections.OrderedDict()
# d2['e'] = 'E'
# d2['d'] = 'D'
# d2['c'] = 'C'
# d2['b'] = 'B'
# d2['a'] = 'A'
#
# print(d1==d2)
#
#
#
#
# # 순전파 결과 출력
# import sys,os
# sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
# import numpy as np
# from common.layers import *
# from common.gradient import numerical_gradient
# from collections import OrderedDict
# import matplotlib.pyplot as plt
# from dataset.mnist import load_mnist
#
# class TwoLayerNet:
#     def __init__(self):
#         # 가중치 초기화
#         self.params = {}
#         self.params['W1'] = np.array([[1,2,3],[4,5,6]]) #(2,3)
#         self.params['b1'] = np.array([1,2,3], ndmin=2) # (2, )
#         self.params['W2'] = np.array([[1,2,3],[4,5,6], [7,8,9]]) #(3,3)
#         self.params['b2'] = np.array([1,2,3], ndmin=2) #(2, )
#
#         # 계층 생성
#         self.layers = OrderedDict()
#         self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
#         self.layers['Relu1'] = Relu()
#         self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
#         self.lastLayer = SoftmaxWithLoss()
#
#     def predict(self, x):
#         for layer in self.layers.values():
#             x = layer.forward(x)
#         return x
#
#     # x : 입력 데이터, t : 정답 레이블
#
#     def loss(self, x, t):
#         y = self.predict(x)
#         return self.lastLayer.forward(y, t)
#
#
#     def accuracy(self, x, t):
#         y = self.predict(x)
#         y = np.argmax(y, axis=1)
#         if t.ndim != 1: t = np.argmax(t, axis=1)
#         accuracy = np.sum(y == t) / float(x.shape[0])
#         return accuracy
#
#     # x : 입력 데이터, t : 정답 레이블
#
#     def gradient(self, x, t):
#         # forward
#         self.loss(x, t)
#
#         # backward
#         dout = 1
#         dout = self.lastLayer.backward(dout)
#         layers = list(self.layers.values())
#         layers.reverse()
#
#         for layer in layers:
#             dout = layer.backward(dout)
#
#         # 결과 저장
#         grads = {}
#         grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
#         grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
#
#         return grads
#
# network = TwoLayerNet()
# x = np.array([[1,2],[3,4],[5,6]])
# t = np.array([[3,4,5], [2,1,4], [2,5,6]])
#
# print(network.predict(x))
#
#
#
#
# # 역전파된 dx 값 출력
# import sys,os
# sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
# import numpy as np
# from common.layers import *
# from common.gradient import numerical_gradient
# from collections import OrderedDict
# import matplotlib.pyplot as plt
# from dataset.mnist import load_mnist
#
# class TwoLayerNet:
#     def __init__(self):
#         # 가중치 초기화
#         self.params = {}
#         self.params['W1'] = np.array([[1,2,3],[4,5,6]]) #(2,3)
#         self.params['b1'] = np.array([1,2,3], ndmin=2) # (2, )
#         self.params['W2'] = np.array([[1,2,3],[4,5,6], [7,8,9]]) #(3,3)
#         self.params['b2'] = np.array([1,2,3], ndmin=2) #(2, )
#
#         # 계층 생성
#         self.layers = OrderedDict()
#         self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
#         self.layers['Relu1'] = Relu()
#         self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
#         self.lastLayer = SoftmaxWithLoss()
#
#     def predict(self, x):
#         for layer in self.layers.values():
#             x = layer.forward(x)
#         return x
#
#     # x : 입력 데이터, t : 정답 레이블
#
#     def loss(self, x, t):
#         y = self.predict(x)
#         return self.lastLayer.forward(y, t)
#
#
#     def accuracy(self, x, t):
#         y = self.predict(x)
#         y = np.argmax(y, axis=1)
#         if t.ndim != 1: t = np.argmax(t, axis=1)
#         accuracy = np.sum(y == t) / float(x.shape[0])
#         return accuracy
#
#     # x : 입력 데이터, t : 정답 레이블
#
#     def gradient(self, x, t):
#         # forward
#         self.loss(x, t)
#
#         # backward
#         dout = 1
#         dout = self.lastLayer.backward(dout)
#         layers = list(self.layers.values())
#         layers.reverse()
#
#         for layer in layers:
#             dout = layer.backward(dout)
#
#             print(layer.__class__.__name__, 'dx :', dout)
#
#         # 결과 저장
#         grads = {}
#         grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
#         grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
#
#         return grads
#
# network = TwoLayerNet()
# x = np.array([[1,2],[3,4],[5,6]])
# t = np.array([[3,4,5], [2,1,4], [2,5,6]])
#
#
# print(network.gradient(x, t))
#
#
#
#
#
# # 오차 역전파로 구현한 mnist 2층 신경망 구현 코드
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)


        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    # x : 입력 데이터, t : 정답 레이블

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    # x : 입력 데이터, t : 정답 레이블

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        return grads

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)


# 하이퍼파라미터
iters_num = 10000  # 반복 횟수를 적절히 설정한다.
train_size = x_train.shape[0] # 60000 개
batch_size = 100  # 미니배치 크기
learning_rate = 0.1
train_loss_list = []
train_acc_list = []
test_acc_list = []


# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)
print(iter_per_epoch) # 600

for i in range(iters_num): # 10000
    # 미니배치 획득  # 랜덤으로 100개씩 뽑아서 10000번을 수행하니까 백만번
    batch_mask = np.random.choice(train_size, batch_size) # 100개 씩 뽑아서 10000번 백만번
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]


    # 기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    # 매개변수 갱신

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]


    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss) # cost 가 점점 줄어드는것을 보려고
    # 1에폭당 정확도 계산 # 여기는 훈련이 아니라 1에폭 되었을때 정확도만 체크

    if i % iter_per_epoch == 0: # 600 번마다 정확도 쌓는다.
        print(x_train.shape) # 60000,784
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

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# import tensorflow as tf
#
#
# def batch_norm(self, input, shape, training, convl=True, name='BN'):
#     beta = tf.Variable(tf.constant(0.0, shape=[shape]), name='beta', trainable=True)
#     scale = tf.Variable(tf.constant(1.0, shape=[shape]), name='gamma', trainable=True)
#     if convl:
#         batch_mean, batch_var = tf.nn.moments(input, [0, 1, 2], name='moments')
#     else:
#         batch_mean, batch_var = tf.nn.moments(input, [0], name='moments')
#
#     ema = tf.train.ExponentialMovingAverage(decay=0.5)
#
#
#     def mean_var_with_update():
#         update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#         with tf.control_dependencies([update_ops]):
#             return tf.identity(batch_mean), tf.identity(batch_var)
#
#     mean, var = tf.cond(training,
#                         mean_var_with_update,
#                         lambda: (ema.average(batch_mean), ema.average(batch_var)))
#     return tf.nn.batch_normalization(input, mean, var, beta, scale, variance_epsilon=1e-3, name=name)
#
#
#
#
#
#
# def batch_norm(self, input, shape, training, convl=True, name='BN'):
#     beta = tf.Variable(tf.constant(0.0, shape=[shape]), name='beta', trainable=True)
#     scale = tf.Variable(tf.constant(1.0, shape=[shape]), name='gamma', trainable=True)
#     if convl:
#         batch_mean, batch_var = tf.nn.moments(input, [0, 1, 2], name='moments')
#     else:
#         batch_mean, batch_var = tf.nn.moments(input, [0], name='moments')
#
#     ema = tf.train.ExponentialMovingAverage(decay=0.5)
#
#     def mean_var_with_update():
#         update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#         if update_ops:
#             updates = tf.group(*update_ops)
#             # total_loss = control_flow_ops.with_dependencies([updates], total_loss)
#
#         with tf.control_dependencies([update_ops]):
#             return tf.identity(batch_mean), tf.identity(batch_var)
#
#     mean, var = tf.cond(training,
#                         mean_var_with_update,
#                         lambda: (ema.average(batch_mean), ema.average(batch_var)))
#     return tf.nn.batch_normalization(input, mean, var, beta, scale, variance_epsilon=1e-3, name=name)
#
#
#
#
#
#
#
#
#
#
#
# #########################################################################################
#
# def batch_norm(self,input, shape, training, convl=True, name='BN'):
#     beta = tf.Variable(tf.constant(0.0, shape=[shape]), name='beta', trainable=True)
#     scale = tf.Variable(tf.constant(1.0, shape=[shape]), name='gamma', trainable=True)
#     if convl:
#         batch_mean, batch_var = tf.nn.moments(input, [0, 1, 2], name='moments')
#     else:
#         batch_mean, batch_var = tf.nn.moments(input, [0], name='moments')
#
#     def mean_var_with_update():
#         update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#         updates = tf.group(*update_ops)
#
#         with tf.control_dependencies([updates]):
#             return tf.identity(batch_mean), tf.identity(batch_var)
#     mean, var = tf.cond(training,
#                          mean_var_with_update,
#                         lambda: (batch_mean, batch_var))
#
#     return tf.nn.batch_normalization(input, mean, var, beta, scale, variance_epsilon=1e-3, name=name)
#
# ###################################################################################################
#
# def batch_norm(self, input, shape, training, convl=True, name='BN'):
#     beta = tf.Variable(tf.constant(0.0, shape=[shape]), name='beta', trainable=True)
#     scale = tf.Variable(tf.constant(1.0, shape=[shape]), name='gamma', trainable=True)
#     if convl:
#         batch_mean, batch_var = tf.nn.moments(input, [0, 1, 2], name='moments')
#     else:
#         batch_mean, batch_var = tf.nn.moments(input, [0], name='moments')
#     ema = tf.train.ExponentialMovingAverage(decay=0.5)
#
#     def mean_var_with_update():
#         ema_apply_op = ema.apply([batch_mean, batch_var])
#         updates = tf.group(*ema_apply_op)
#         with tf.control_dependencies([updates]):
#             return tf.identity(batch_mean), tf.identity(batch_var)
#     mean, var = tf.cond(training,
#                         mean_var_with_update,
#                         lambda: (batch_mean, batch_var))
#     return tf.nn.batch_normalization(input, mean, var, beta, scale, variance_epsilon=1e-3, name=name)


import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist

class TwoLayerNet:
    def __init__(self, input_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, output_size)
        self.params['b1'] = np.zeros(output_size)
        # self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        # self.params['b2'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        # self.layers['Relu1'] = Relu()
        # self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        # self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        print(self.layers)
        for layer in self.layers.values():
            x = layer.forward(x)
            print(x)
        return x
    # x : 입력 데이터, t : 정답 레이블

    def loss(self, x, t):
        y = self.predict(x)
        return self.layers.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    # x : 입력 데이터, t : 정답 레이블

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        # grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        # grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        # backward
        dout = 1
        dout = self.layers.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        # grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        return grads

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, output_size=10)


# 하이퍼파라미터
iters_num = 10000  # 반복 횟수를 적절히 설정한다.
train_size = x_train.shape[0] # 60000 개
batch_size = 100  # 미니배치 크기
learning_rate = 0.1
train_loss_list = []
train_acc_list = []
test_acc_list = []


# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)
print(iter_per_epoch) # 600

for i in range(iters_num): # 10000
    # 미니배치 획득  # 랜덤으로 100개씩 뽑아서 10000번을 수행하니까 백만번
    batch_mask = np.random.choice(train_size, batch_size) # 100개 씩 뽑아서 10000번 백만번
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]


    # 기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    # 매개변수 갱신

    for key in ('W1', 'b1'):
        network.params[key] -= learning_rate * grad[key]


    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss) # cost 가 점점 줄어드는것을 보려고
    # 1에폭당 정확도 계산 # 여기는 훈련이 아니라 1에폭 되었을때 정확도만 체크

    if i % iter_per_epoch == 0: # 600 번마다 정확도 쌓는다.
        print(x_train.shape) # 60000,784
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc) # 10000/600 개  16개 # 정확도가 점점 올라감
        test_acc_list.append(test_acc)  # 10000/600 개 16개 # 정확도가 점점 올라감
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

import sys, os

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, hidden_size)
        self.params['b2'] = np.zeros(output_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    # x : 입력 데이터, t : 정답 레이블

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x : 입력 데이터, t : 정답 레이블

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        grads['W3'] = numerical_gradient(loss_W, self.params['W3'])
        grads['b3'] = numerical_gradient(loss_W, self.params['b3'])
        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        grads['W3'], grads['b3'] = self.layers['Affine3'].dW, self.layers['Affine3'].db
        return grads


# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 하이퍼파라미터
iters_num = 10000  # 반복 횟수를 적절히 설정한다.
train_size = x_train.shape[0]  # 60000 개
batch_size = 100  # 미니배치 크기
learning_rate = 0.1
train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)
print(iter_per_epoch)  # 600

for i in range(iters_num):  # 10000
    # 미니배치 획득  # 랜덤으로 100개씩 뽑아서 10000번을 수행하니까 백만번
    batch_mask = np.random.choice(train_size, batch_size)  # 100개 씩 뽑아서 10000번 백만번
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    # grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    # 매개변수 갱신

    for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
        network.params[key] -= learning_rate * grad[key]

    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)  # cost 가 점점 줄어드는것을 보려고
    # 1에폭당 정확도 계산 # 여기는 훈련이 아니라 1에폭 되었을때 정확도만 체크

    if i % iter_per_epoch == 0:  # 600 번마다 정확도 쌓는다.
        print(x_train.shape)  # 60000,784
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)  # 10000/600 개  16개 # 정확도가 점점 올라감
        test_acc_list.append(test_acc)  # 10000/600 개 16개 # 정확도가 점점 올라감
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))








import sys,os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict
from dataset.mnist import load_mnist as mnist
import numpy as np

class TwoLayerNet:
    def __init__(self,input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    # x : 입력 데이터, t : 정답 레이블

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)


    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x : 입력 데이터, t : 정답 레이블

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        grads['W3'], grads['b3'] = self.layers['Affine3'].dW, self.layers['Affine3'].db
        return grads

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
(x_train, t_train), (x_test, t_test) = mnist(normalize=True, one_hot_label=True)

iters_num = 10000  # 반복 횟수를 적절히 설정한다.
train_size = x_train.shape[0] # 60000 개
batch_size = 100  # 미니배치 크기
learning_rate = 0.1
train_loss_list = []
train_acc_list = []
test_acc_list = []


# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)
print(iter_per_epoch) # 600

for i in range(iters_num): # 10000
    # 미니배치 획득  # 랜덤으로 100개씩 뽑아서 10000번을 수행하니까 백만번
    batch_mask = np.random.choice(train_size, batch_size) # 100개 씩 뽑아서 10000번 백만번
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]


    # 기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    # 매개변수 갱신

    for key in ('W1', 'b1','W2','b2'):
        network.params[key] -= learning_rate * grad[key]


    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss) # cost 가 점점 줄어드는것을 보려고
    # 1에폭당 정확도 계산 # 여기는 훈련이 아니라 1에폭 되었을때 정확도만 체크

    if i % iter_per_epoch == 0: # 600 번마다 정확도 쌓는다.
        print(x_train.shape) # 60000,784
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc) # 10000/600 개  16개 # 정확도가 점점 올라감
        test_acc_list.append(test_acc)  # 10000/600 개 16개 # 정확도가 점점 올라감
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))









import sys,os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict
from dataset.mnist import load_mnist as mnist
import numpy as np

class TwoLayerNet:
    def __init__(self,input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, output_size)
        self.params['b1'] = np.zeros(output_size)
        # self.params['W2'] = weight_init_std * np.random.randn(hidden_size, hidden_size)
        # self.params['b2'] = np.zeros(hidden_size)
        # self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        # self.params['b3'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        # self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        # self.layers['Relu2'] = Relu()
        # self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    # x : 입력 데이터, t : 정답 레이블

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)


    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x : 입력 데이터, t : 정답 레이블

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        # grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        # grads['W3'], grads['b3'] = self.layers['Affine3'].dW, self.layers['Affine3'].db
        return grads

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
(x_train, t_train), (x_test, t_test) = mnist(normalize=True, one_hot_label=True)

iters_num = 10000  # 반복 횟수를 적절히 설정한다.
train_size = x_train.shape[0] # 60000 개
batch_size = 100  # 미니배치 크기
learning_rate = 0.1
train_loss_list = []
train_acc_list = []
test_acc_list = []


# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)
print(iter_per_epoch) # 600

for i in range(iters_num): # 10000
    # 미니배치 획득  # 랜덤으로 100개씩 뽑아서 10000번을 수행하니까 백만번
    batch_mask = np.random.choice(train_size, batch_size) # 100개 씩 뽑아서 10000번 백만번
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]


    # 기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    # 매개변수 갱신

    for key in ('W1', 'b1'):
        network.params[key] -= learning_rate * grad[key]


    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss) # cost 가 점점 줄어드는것을 보려고
    # 1에폭당 정확도 계산 # 여기는 훈련이 아니라 1에폭 되었을때 정확도만 체크

    if i % iter_per_epoch == 0: # 600 번마다 정확도 쌓는다.
        print(x_train.shape) # 60000,784
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc) # 10000/600 개  16개 # 정확도가 점점 올라감
        test_acc_list.append(test_acc)  # 10000/600 개 16개 # 정확도가 점점 올라감
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))