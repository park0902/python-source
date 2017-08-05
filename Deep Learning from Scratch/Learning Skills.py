'''
--------------------------------------------------------------------------------------
- 확률적 경사 하강법(Stochastic Gradient Descent)

                 aL
    W <- W - n ------       W : 갱신할 가중치 매개변수
                 aW         
                           aL
                          ---- : W 에 대한 손실함수의 기울기
                           aW
                           
                            n : 학습률(실제로는 0.01 나 0.001 과 같은 값을 미리 정해서 사용)
                            
                            
    SGD 단점 : 비등방성(anisotropy) 함수(방향에 따라 성질, 즉 여기에서는 기울기가 달라지는 함수)에서는 탐색 경로가 비효율적!!
--------------------------------------------------------------------------------------
'''

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.key():
            params[key] -= self.lr * grads[key]




'''
--------------------------------------------------------------------------------------
- 모멘텀

    운동량을 뜻하는 단어로 물리와 관계가 있다
    
                  aL
    v <- av - n ------        W : 갱신할 가중치 매개변수
                  aW
                            aL
                           ---- : W 에 대한 손실함수의 기울기
    W <- W + v              aW      
                                                               
                             n : 학습률(실제로는 0.01 나 0.001 과 같은 값을 미리 정해서 사용)       
                             
                             v : 속도
                             
                          av항 : 물체가 아무런 힘을 받지 않을 때 서서히 하강시키는 역할 (a는 0.9 등의 값으로 설정)  
--------------------------------------------------------------------------------------
'''

import numpy as np

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)    # 해당 변수의 구조대로 생성

        for key in params.key():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]




'''
--------------------------------------------------------------------------------------
- AdaGrad

    각각의 매개변수에 맞게 맞춤형 값을 만들어준다!
    개별 매개변수에 적응적으로 학습률을 조정하면서 학습을 진행
    

    신경망 학습에서는 학습률(수식에서는 η 으로 표시) 값이 중요!
    
    이 값이 너무 작으면 학습 시간이 너무 길어지고, 반대고 너무 크면 발산하여 학습이 제대로 이뤄지지 않는다
    
    학습률 감소(learning rate decay) : 학습을 진행하면서 학습률을 점차 줄여가는 방법
    
               aL        aL                              1
    h <- h + ------  * ------        매개변수를 갱신할 때 ------ 을 곱해서 학습률 조정!
               aW        aW                            루트 h
               
                  1        aL
    W <- W - η ------- * ------
                루트 h      aW
                
    매개변수의 원소 중에서 많이 움직인(크게 갱신된) 원소는 학습률이 낮아진다는 뜻인데,
    다시 말해서 학습률 감소가 매개변수의 원소마다 다르게 적용됨!
--------------------------------------------------------------------------------------
'''

class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, parames, grads):
        if self.h is None:
            self.h = {}
            for key, val in parames.items():
                self.h[key] = np.zeros_like(val)

        for key in parames.key():
            self.h[key] += grads[key] * grads[key]
            parames[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)





import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)


input_data = np.random.randn(1000, 100)  # 1000개의 데이터
node_num = 100  # 각 은닉층의 노드(뉴런) 수
hidden_layer_size = 5  # 은닉층이 5개
activations = {}  # 이곳에 활성화 결과를 저장

x = input_data

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i - 1]

    # 초깃값을 다양하게 바꿔가며 실험해보자！
    w = np.random.randn(node_num, node_num) * 1
    # w = np.random.randn(node_num, node_num) * 0.01
    # w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)
    # w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)


    a = np.dot(x, w)

    # 활성화 함수도 바꿔가며 실험해보자！
    z = sigmoid(a)
    # z = ReLU(a)
    # z = tanh(a)

    activations[i] = z

# 히스토그램 그리기
for i, a in activations.items():
    plt.subplot(1, len(activations), i + 1)
    plt.title(str(i + 1) + "-layer")
    if i != 0: plt.yticks([], [])
    # plt.xlim(0.1, 1)
    # plt.ylim(0, 7000)
    plt.hist(a.flatten(), 30, range=(0, 1))
plt.show()
print(activations)







import os
import sys

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from MNIST import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 오버피팅을 재현하기 위해 학습 데이터 수를 줄임
x_train = x_train[:300]
t_train = t_train[:300]

# weight decay（가중치 감쇠） 설정 =======================
#weight_decay_lambda = 0 # weight decay를 사용하지 않을 경우
weight_decay_lambda = 0.1
# ====================================================

network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10,
                        weight_decay_lambda=weight_decay_lambda)
optimizer = SGD(lr=0.01) # 학습률이 0.01인 SGD로 매개변수 갱신

max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(1000000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print("epoch:" + str(epoch_cnt) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc))

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break


# 그래프 그리기==========
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
