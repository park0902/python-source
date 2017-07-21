# # coding: utf-8
# import sys, os
# sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
# import numpy as np
# import pickle               # mnist 데이터셋을 읽어올때 인터넷이 연결되어 있는 상태에서 가져와야하는데 이때 시간이 걸린다
#                             # 가져온 이미지를 로컬에 저장할때 pickle 파일로 생성이 되고 로컬에 저장되어있으면 순식간에 읽을수 있다
#
# from dataset.mnist import load_mnist
# from common.functions import sigmoid, softmax
#
#
# def get_data():
#     (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
#     return x_test, t_test
#
# '''
# MNIST를 위한 신경망 설계
#
#  •입력층 : 784개 (이미지 크기가 28x28 픽셀임으로)
#  •출력층 : 10개 (0에서 9로 분류함으로)
#  •첫번째 은닉층 : 50개 (임의의 수)
#  •두번째 은닉층 : 100개 (임의의 수)
# '''
#
#
#
# def init_network():
#     with open("sample_weight.pkl", 'rb') as f:
#         network = pickle.load(f)
#     return network
#
#
# def predict(network, x):
#     W1, W2, W3 = network['W1'], network['W2'], network['W3']
#     b1, b2, b3 = network['b1'], network['b2'], network['b3']
#
#     a1 = np.dot(x, W1) + b1
#     z1 = sigmoid(a1)
#     a2 = np.dot(z1, W2) + b2
#     z2 = sigmoid(a2)
#     a3 = np.dot(z2, W3) + b3
#     y = softmax(a3)
#
#     return y
#
#
# x, t = get_data()
# network = init_network()
# accuracy_cnt = 0
# for i in range(len(x)):
#     y = predict(network, x[i])
#     p= np.argmax(y) # 확률이 가장 높은 원소의 인덱스를 얻는다.
#     if p == t[i]:
#         accuracy_cnt += 1
#
# print("Accuracy:" + str(float(accuracy_cnt) / len(x)))




#
# coding: utf-8
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

