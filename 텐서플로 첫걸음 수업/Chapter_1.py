# # 텐서플로우를 이용하지 않고 그냥 파이썬으로 단층 신경망 구현
# # coding: utf-8
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
# #    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
#     def __init__(self, input_size,  output_size, weight_init_std=0.01):
#         # 가중치 초기화
#         self.params = {}
#         self.params['W1'] = weight_init_std * np.random.randn(input_size, output_size)
#         self.params['b1'] = np.zeros(output_size)
#        # self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
#        # self.params['b2'] = np.zeros(output_size)
#         # 계층 생성
#         self.layers = OrderedDict()
#         self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
#         self.layers['Relu1'] = Relu()
#       #  self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
#         self.lastLayer = SoftmaxWithLoss()
#
#
#     def predict(self, x):
#         for layer in self.layers.values():
#             x = layer.forward(x)
#         return x
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
#     # x : 입력 데이터, t : 정답 레이블
#
#     def numerical_gradient(self, x, t):
#         loss_W = lambda W: self.loss(x, t)
#         grads = {}
#         grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
#         grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
#         grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
#         grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
#         return grads
#
#
#     def gradient(self, x, t):
#         # forward
#         self.loss(x, t)
#         # backward
#         dout = 1
#         dout = self.lastLayer.backward(dout)
#         layers = list(self.layers.values())
#         layers.reverse()
#         for layer in layers:
#             dout = layer.backward(dout)
#         # 결과 저장
#         grads = {}
#         grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
#       #  grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
#         return grads
#
# # 데이터 읽기
# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# #network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
# network = TwoLayerNet(input_size=784,output_size=10)
#
# # 하이퍼파라미터
# iters_num = 10000  # 반복 횟수를 적절히 설정한다.
# train_size = x_train.shape[0] # 60000 개
# batch_size = 100  # 미니배치 크기
# learning_rate = 0.1
# train_loss_list = []
# train_acc_list = []
# test_acc_list = []
#
#
# # 1에폭당 반복 수
# iter_per_epoch = max(train_size / batch_size, 1)
# print(iter_per_epoch) # 600
#
# for i in range(iters_num): # 10000
#     # 미니배치 획득  # 랜덤으로 100개씩 뽑아서 10000번을 수행하니까 백만번
#     batch_mask = np.random.choice(train_size, batch_size) # 100개 씩 뽑아서 10000번 백만번
#     x_batch = x_train[batch_mask]
#     t_batch = t_train[batch_mask]
#
#
#
#     # 기울기 계산
#     #grad = network.numerical_gradient(x_batch, t_batch)
#     grad = network.gradient(x_batch, t_batch)
#     # 매개변수 갱신
#
#     for key in ('W1', 'b1'):
#         network.params[key] -= learning_rate * grad[key]
#
#
#     # 학습 경과 기록
#     loss = network.loss(x_batch, t_batch)
#     train_loss_list.append(loss) # cost 가 점점 줄어드는것을 보려고
#     # 1에폭당 정확도 계산 # 여기는 훈련이 아니라 1에폭 되었을때 정확도만 체크
#
#     if i % iter_per_epoch == 0: # 600 번마다 정확도 쌓는다.
#         print(x_train.shape) # 60000,784
#         train_acc = network.accuracy(x_train, t_train)
#         test_acc = network.accuracy(x_test, t_test)
#         train_acc_list.append(train_acc) # 10000/600 개  16개 # 정확도가 점점 올라감
#         test_acc_list.append(test_acc)  # 10000/600 개 16개 # 정확도가 점점 올라감
#         print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
#
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



# 단층 신경망을 텐서플로우로 구현
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


import tensorflow as tf

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

x = tf.placeholder("float", [None, 784])
y = tf.nn.softmax(tf.matmul(x,W) + b)       # todo     Afiine 한 결과를 softmax 함수에 바로 입력해서 한번에 수행되고 예상값 리턴
y_ = tf.placeholder("float", [None,10])     # todo     교차엔트로피를 구현하기 위해서 실제 레이블을 담고있는 새로운 플레이스 홀더를 생성

cross_entropy = -tf.reduce_sum(y_*tf.log(y))    # todo  비용함수를 구현하는데 여기서 사용되는 reduce_sum 은 차원축소후 sum 하는 함수
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)    # todo  lr=0.01 과 SGD 경사하강법으로 비용함수 의 오차가
                                                                                # todo  최소화 되겠금 역전파 시킴

sess = tf.Session()                             # todo  텐서플로우 그래프 연산을 시작하겠금 session 객체 생성
sess.run(tf.global_variables_initializer())     # todo  모든 변수 초기화

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)    # todo  훈련데이터셋에서 무작위로 100개 추출
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})     # todo  100개 데이터를 SGD 의 경사감소법으로 훈련시킨다
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))  # todo  y 라벨(예상)중 가장 큰 인덱스 와 y_라벨(실제) 중 가장 큰 인덱스 리턴해서 같은지 비교
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) # todo  [True, False, True] -> [1, 0, 1] 로 변경해주고 평균 출력
    if i % 100 == 0:
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))



# reduce_sum 과 reduce_mean
import numpy as np
import tensorflow as tf

x = np.arange(6).reshape(2,3)
print(x)

sess = tf.Session()
print(sess.run(tf.reduce_sum(x)))
print(sess.run(tf.reduce_sum(x, 0)))    # 열 단위
print(sess.run(tf.reduce_sum(x, 1)))    # 행 단위



# 숫자0으로 채워진 2행3열의 행렬을 만들고 숫자1로 채워진 행렬을 만들고 두 행렬의 합 구하기
import tensorflow as tf

a = tf.zeros([2,3])
b = tf.ones([2,3])

sess = tf.Session()
print(sess.run(tf.add(a,b)))



# 숫자2로 채워진 2행 3열의 행렬을 만들고 숫자3으로 채워진 2행 3열의 행렬을 만들고 두 행렬의 합 출력
import tensorflow as tf

a = tf.placeholder("float", shape=[2,3])
b = tf.placeholder("float", shape=[2,3])
result = tf.add(a,b)

sess = tf.Session()

# print(sess.run(a, feed_dict={a: [[2,2,2],[2,2,2]]}))
# print(sess.run(b, feed_dict={b: [[3,3,3],[3,3,3]]}))
print(sess.run(result, feed_dict={a: [[2,2,2],[2,2,2]],
                                  b: [[3,3,3],[3,3,3]]}))



# 숫자2로 채워진 2X3 행렬과 숫자3으로 채워진 3X2 행렬의 행렬곱 출력
import tensorflow as tf

a = tf.placeholder("float", shape=[2,3])
b = tf.placeholder("float", shape=[3,2])
result = tf.matmul(a,b)

sess = tf.Session()

print(sess.run(result, feed_dict={a: [[2,2,2],[2,2,2]],
                                  b: [[3,3],[3,3],[3,3]]}))




# 아래의 진리 연산 True 를 1로 False 를 0으로 출력
import tensorflow as tf

correct_prediction = [ True, False , True  ,True  ,True  ,True  ,True,  True  ,True  ,True  ,True  ,True
  ,True  ,True  ,True, False , True  ,True, False , True  ,True  ,True  ,True  ,True
  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True
  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True,
  True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True
  ,True  ,True  ,True  ,True  ,True  ,True ,False , True  ,True  ,True  ,True  ,True
  ,True  ,True, False , True, False , True  ,True  ,True  ,True  ,True  ,True  ,True
  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True
 ,False , True  ,True  ,True]

sess = tf.Session()

accuracy = tf.cast(correct_prediction, "float")

print(sess.run(accuracy))




# 위에서 출력한 100개의 숫자의 평균값 출력
import tensorflow as tf

correct_prediction = [ True, False , True  ,True  ,True  ,True  ,True,  True  ,True  ,True  ,True  ,True
  ,True  ,True  ,True, False , True  ,True, False , True  ,True  ,True  ,True  ,True
  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True
  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True,
  True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True
  ,True  ,True  ,True  ,True  ,True  ,True ,False , True  ,True  ,True  ,True  ,True
  ,True  ,True, False , True, False , True  ,True  ,True  ,True  ,True  ,True  ,True
  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True
 ,False , True  ,True  ,True]

sess = tf.Session()

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(sess.run(accuracy))




# 다른 데이터 예제
import tensorflow as tf
import numpy as np
from dataset.mnist import load_mnist

##### mnist 데이터 불러오기 및 정제 #####

############################################
# mnist 데이터 중 10000개 저장
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, one_hot_label=True)
input = np.concatenate((x_train, x_test), axis=0)
target = np.concatenate((t_train, t_test), axis=0)
print('input shape :', input.shape, '| target shape :', target.shape)
a = np.concatenate((input, target), axis=1)
np.savetxt('mnist.csv', a[:10000], delimiter=',')
print('mnist.csv saved')
############################################

# 파일 로드 및 변수 설정
save_status = True
load_status = True

mnist = np.loadtxt('mnist.csv', delimiter=',', unpack=False, dtype='float32')
print('mnist.csv loaded')
print('mnist shape :',mnist.shape)

train_num = int(mnist.shape[0] * 0.8)

x_train, x_test = mnist[:train_num,:784], mnist[train_num:,:784]
t_train, t_test = mnist[:train_num,784:], mnist[train_num:,784:]

print('x train shape :',x_train.shape, '| x target shape :',x_test.shape)
print('t train shape :',t_train.shape, '| t target shape :',t_test.shape)

global_step = tf.Variable(0, trainable=False, name='global_step')
X = tf.placeholder(tf.float32,[None, 784])
T = tf.placeholder(tf.float32,[None, 10])
W = tf.Variable(tf.random_uniform([784,10], -1e-7, 1e-7)) # [784,10] 형상을 가진 -1e-7 ~ 1e-7 사이의 균등분포 어레이
b = tf.Variable(tf.random_uniform([10], -1e-7, 1e-7))    # [10] 형상을 가진 -1e-7 ~ 1e-7 사이의 균등분포 벡터
Y = tf.add(tf.matmul(X,W), b) # tf.matmul(X,W) + b 와 동일

############################################
# 그외 가중치 초기화 방법
# W = tf.Variable(tf.random_uniform([784,10], -1, 1)) # [784,10] 형상을 가진 -1~1 사이의 균등분포 어레이
# W = tf.get_variable(name="W", shape=[784, 10], initializer=tf.contrib.layers.xavier_initializer()) # xavier 초기값
# W = tf.get_variable(name='W', shape=[784, 10], initializer=tf.contrib.layers.variance_scaling_initializer()) # he 초기값
# b = tf.Variable(tf.zeros([10]))
############################################

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=T, logits=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.05).minimize(cost, global_step=global_step)

############################################
# 그외 옵티마이저
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
# optimizer = tf.train.MomentumOptimizer(learning_rate=0.01)
############################################

##### mnist 학습시키기 #####
# 일반 버전
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#todo 로드 버전
# sess = tf.Session()
# saver = tf.train.Saver(tf.global_variables())
#
# cp = tf.train.get_checkpoint_state('./save') # save 폴더를 checkpoint로 설정
# # checkpoint가 설정되고, 폴더가 실제로 존재하는 경우 restore 메소드로 변수, 학습 정보 불러오기
# if cp and tf.train.checkpoint_exists(cp.model_checkpoint_path):
#     saver.restore(sess, cp.model_checkpoint_path)
#     print(sess.run(global_step),'회 학습한 데이터 로드 완료')
# # 그렇지 않은 경우 일반적인 sess.run()으로 tensorflow 실행
# else:
#     sess.run(tf.global_variables_initializer())
#     print('새로운 학습 시작')

# epoch, batch 설정
epoch = 100
total_size = x_train.shape[0]
batch_size = 100
# mini_batch_size = 100
total_batch = int(total_size/batch_size)

# 정확도 계산 함수
correct_prediction = tf.equal(tf.argmax(T, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 설정한 epoch 만큼 루프
for each_epoch in range(epoch):
    total_cost = 0
    # 각 epoch 마다 batch 크기만큼 데이터를 뽑아서 학습
    for idx in range(0, total_size, batch_size):
        batch_x, batch_y = x_train[idx:idx+batch_size], t_train[idx:idx+batch_size]

        _, cost_val = sess.run([optimizer, cost], feed_dict={X : batch_x, T : batch_y})
        total_cost += cost_val

    print('Epoch:', '%04d' % (each_epoch + 1),
          'Avg. cost =', '{:.8f}'.format(total_cost / total_batch),
          )

print('최적화 완료!')

#todo 최적화가 끝난 뒤, 변수와 학습 정보 저장
# saver.save(sess, './save/mnist_dnn.ckpt', global_step=global_step)

##### 학습 결과 확인 #####
print('Train 정확도 :', sess.run(accuracy, feed_dict={X: x_train, T: t_train}))
print('Test 정확도:', sess.run(accuracy, feed_dict={X: x_test, T: t_test}))


# 문제1. 1) 가중치 초기값을 xavier 초기값으로 설정, 2) 옵티마이저를 momentum 옵티마이저로 설정 후, 3) epoch은 200번,
#       4) batch_size 는 200으로 수정하여 학습해보기

# 문제2. 19~20번째 줄의 save_status 와 load_status가 각각 True 인 경우에만 저장/불러오기 되도록 코드 수정

# 문제3. 82번째 줄의 mini_batch_size를 이용하여 200개의 배치 데이터 중 100개만 랜덤으로 뽑아 학습하도록 코드 수정
#       (힌트 : np.random.randint(low=a, high=b, size=c) --> 숫자 a~b 사이의 정수 c개를 랜덤으로 뽑아주는 함수)

# 문제4. 훈련데이터의 10%를 뽑아 만든 검증 데이터로 아래 형식과 같이 50번째 epoch 마다 정확도 출력해보기.
#         (훈련데이터(0.9) + 검증데이터(0.1) = 전체의 80%   /   테스트 데이터 = 전체의 20%)
'''
Epoch: 0048 Avg. cost = 0.21684368
Epoch: 0049 Avg. cost = 0.16016438
Epoch: 0050 Avg. cost = 0.24727620
=================================
50번째 검증 데이터 정확도 : 0.881
=================================
Epoch: 0051 Avg. cost = 0.22011646
Epoch: 0052 Avg. cost = 0.17041421
Epoch: 0053 Avg. cost = 0.13844220
'''



#
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import tensorflow as tf

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

x_image = tf.reshape(x, [-1,28,28,1])
print("x_image=", x_image)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

test_x, test_y = mnist.test.next_batch(1000)
print("test accuracy %g"% sess.run(
        accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))








# 다중 신경망 구현
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf

x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='input')
y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='output')
lr = tf.placeholder(dtype=tf.bool, name='learning_rate')
dropout = tf.Variable(tf.constant(0.5), name='dropout')
with tf.variable_scope('model'):
    with tf.name_scope('Layer1') as scope:
        w1 = tf.get_variable(name='weight1', shape=[784, 800], dtype=tf.float32,
                             initializer=tf.contrib.layers.variance_scaling_initializer())
        b1 = tf.Variable(tf.constant(0.001, shape=[800]), name='b1')
        L_fn1 = tf.nn.xw_plus_b(x=x, weights=w1, biases=b1, name='fnLayer1')
        L_fn1 = tf.nn.relu(L_fn1, name='fn1_Relu')
        L_fn1 = tf.layers.dropout(inputs=L_fn1, rate=dropout, training=lr)

    with tf.name_scope('Layer2') as scope:
        w2 = tf.get_variable(name='weight2', shape=[800, 800], dtype=tf.float32,
                             initializer=tf.contrib.layers.variance_scaling_initializer())
        b2 = tf.Variable(tf.constant(0.001, shape=[800]), name='b2')
        L_fn2 = tf.nn.xw_plus_b(x=L_fn1, weights=w2, biases=b2, name='fnLayer2')
        L_fn2 = tf.nn.relu(L_fn2, name='fn2_Relu')
        L_fn2 = tf.layers.dropout(inputs=L_fn2, rate=dropout, training=lr)

    w_out = tf.get_variable(name='w_out', shape=[800, 10], dtype=tf.float32,
                            initializer=tf.contrib.layers.variance_scaling_initializer())
    b_out = tf.Variable(tf.constant(0.001, shape=[10]), name='b_out')

    logits = tf.nn.xw_plus_b(x=L_fn2, weights=w_out, biases=b_out, name='logits')
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)

    # todo Adam
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    # todo Adagrad
    # optimizer = tf.train.AdagradOptimizer(learning_rate=0.001).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run([loss, optimizer], feed_dict={x: batch_x, y: batch_y, lr: True})
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1)), dtype=tf.float32))
    if i % 100 == 0:
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, lr: False}))



# todo 결과
# Adam
# 0.2239
# 0.9249
# 0.9353
# 0.9487
# 0.9578
# 0.9573
# 0.9614
# 0.9653
# 0.9675
# 0.9708


# Adagrad
# 0.1642
# 0.8692
# 0.8906
# 0.9029
# 0.9078
# 0.9127
# 0.9157
# 0.9183
# 0.92
# 0.9227



# 선형회귀
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_points = 1000
vectors_set = []
for i in range(num_points):
    x1 = np.random.normal(0.0,0.55) # np.random.norminal(mean,sd,size)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0,0.03)
    vectors_set.append([x1,y1])

# range: 결과값 list형태  xrange
# xrange : 결과값 xrange형태
# 지정한 데이터 크기에 상관없이 메모리 할당량이 일정(지정하는 범위가 클 수록 xrange 사용이 메모리 측면에서 효율적
# 자신이 속한 데이터 값을 한꺼번에 메모리에 로드하는 것이 아니라 해당 값에 접근할 때마다 그 값을 하나씩 로딩(generator의 yield와 동일한 효과)
# 3.x 버전대에서는 지원 x

x_data = [v[0] for v in vectors_set] # ex)[-0.5103009977197533, -0.381074944615835,...]
y_data = [v[1] for v in vectors_set] # ex)[0.2610048258762149, 0.2735388643503902,...]

# 그래프 표시
plt.plot(x_data, y_data, 'ro') #r:red, o:circle
# https://rstockanalysis.blogspot.kr/2017/03/plot-with-python-1.html 참고
plt.show()

# 선형 모델 정의
W = tf.Variable(tf.random_uniform([1],-1.0, 1.0)) #tf.random_uniform(shape, minval, maxval)
b = tf.Variable(tf.zeros([1])) #tf.zeros(shape) 해당 shape로 0
y = W * x_data + b

#손실함수 정의
loss = tf.reduce_mean(tf.square(y-y_data)) #MSE
optimizer = tf.train.GradientDescentOptimizer(0.5) # 경사하강법으로 loss func 최소화, 0.5는 learning rate

train = optimizer.minimize(loss)

init = tf.global_variables_initializer() #모든 변수 초기화

# 세션 시작
sess = tf.Session()
sess.run(init)

# 8번에 걸쳐 학습
for step in range(8):
    sess.run(train)
    print(step, sess.run(W), sess.run(b))
    print(step, sess.run(loss))

    #그래픽 표시
    plt.subplot(4,2,step+1)
    plt.plot(x_data, y_data, 'ro')
    plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
    plt.xlabel('x')
    plt.xlim(-2,2)
    plt.ylim(0.1,0.6)
    plt.ylabel('y')
plt.show()





# 첼린저호 회귀분석
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

data= np.loadtxt("D://data//challenger.csv",delimiter=',', skiprows=1)
x_data = list(data[:,-3])
y_data = list(data[:,-4])

num_points = 1000
vectors_set = []
for i in range(num_points):
    # x1 = np.random.normal(0.0,0.55) # np.random.norminal(mean,sd,size)
    # y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0,0.03)
    vectors_set.append([x_data,y_data])

# range: 결과값 list형태  xrange
# xrange : 결과값 xrange형태
# 지정한 데이터 크기에 상관없이 메모리 할당량이 일정(지정하는 범위가 클 수록 xrange 사용이 메모리 측면에서 효율적
# 자신이 속한 데이터 값을 한꺼번에 메모리에 로드하는 것이 아니라 해당 값에 접근할 때마다 그 값을 하나씩 로딩(generator의 yield와 동일한 효과)
# 3.x 버전대에서는 지원 x

x_data = [v[0] for v in vectors_set] # ex)[-0.5103009977197533, -0.381074944615835,...]
y_data = [v[1] for v in vectors_set] # ex)[0.2610048258762149, 0.2735388643503902,...]

# 그래프 표시
plt.plot(x_data[1], y_data[1], 'ro') #r:red, o:circle
# https://rstockanalysis.blogspot.kr/2017/03/plot-with-python-1.html 참고
plt.show()

# 선형 모델 정의
W = tf.Variable(tf.random_uniform([1],-1.0, 1.0)) #tf.random_uniform(shape, minval, maxval)
b = tf.Variable(tf.zeros([1])) #tf.zeros(shape) 해당 shape로 0
y = W * x_data + b

#손실함수 정의
loss = tf.reduce_mean(tf.square(y-y_data)) #MSE
optimizer = tf.train.GradientDescentOptimizer(0.5) # 경사하강법으로 loss func 최소화, 0.5는 learning rate

train = optimizer.minimize(loss)

init = tf.global_variables_initializer() #모든 변수 초기화

# 세션 시작
sess = tf.Session()
sess.run(init)

# 8번에 걸쳐 학습
for step in range(8):
    sess.run(train)
    print(step, sess.run(W), sess.run(b))
    print(step, sess.run(loss))

    #그래픽 표시
    plt.subplot(4,2,step+1)
    plt.plot(x_data[1], y_data[1], 'ro')
    plt.plot(x_data[1], sess.run(W) * x_data[1] + sess.run(b))
    plt.xlabel('x')
    plt.xlim(-2,2)
    plt.ylim(0.1,0.6)
    plt.ylabel('y')
plt.show()






###################################################################################################
## ▣ K-평균 알고리즘
##  - 주어진 데이터를 지정된 군집 개수(K)로 그룹화 한다.
##    한 군집 내의 데이터들은 동일한 성질을 가지며 다른 그룹과는 구별된다.
##    알고리즘의 결과는 중심이라고 부르는 K개의 점으로서, 이들은 각기 다른 그룹의 중심점을 나타내며 데이터들은
##    K개의 군집 중 하나에만 속할 수 있다.
##    한 군집 내의 모든 데이터들은 다른 어떤 중심들보다 자기 군집 중심과의 거리가 더 가깝다.
##  - 반복 개선 기법 사용
##   1. 초기 단계(0 단계) : K개 중심의 초기 집합을 결정 (K 개를 임의로 선택)
##   2. 할당 단계(1 단계) : 각 데이터를 가장 가까운 군집에 할당
##   3. 업데이트 단계(2 단계) : 각 그룹에 대해 새로운 중심을 계산
###################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

num_points = 2000
vectors_set = []

for i in range(num_points):
    if np.random.random() > 0.5:
        vectors_set.append([np.random.normal(0.0, 0.9), np.random.normal(0.0, 0.9)])
    else:
        vectors_set.append([np.random.normal(3.0, 0.5), np.random.normal(1.0, 0.5)])

# df = pd.DataFrame({'x': [v[0] for v in vectors_set],
#                    'y': [v[1] for v in vectors_set]})
x = [v[0] for v in vectors_set]
y = [v[1] for v in vectors_set]
# plt.plot(x, y, 'ro')
# plt.show()

vectors = tf.constant(vectors_set)
k = 4

#  1단계 : 랜덤으로 K 개의 초기 집합을 설정
#   - tf.random_shuffle(object) : 상수 텐서를 랜덤으로 섞는다.
#   - tf.slice(object, begin, size) : 랜덤으로 얻어진 상수 텐서를 처음부터 k개 만큼 자른다.
#   - tf_Variable(object) : 상수 텐서를 변수 텐서로 변환.
centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0, 0], [k, -1]))

#   - expanded_vectors   : (2000, 2) --> (1, 2000, 2), expanded_vectors.shape
#   - expanded_centroids : (4, 2) --> (4, 1, 2), expanded_centroids.shape
#   ※ 크기가 1인 차원만 broadcasting 기능이 동작하므로, 두 텐서간의 차원 확장이 필요하다.
expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroids = tf.expand_dims(centroids, 1)

#  2단계 : 각 데이터를 가장 가까운 군집에 할당
#  → 중심과 각 점간의 거리를 구하기 위해 유클리드 제곱 거리 알고리즘을 선택.
#   - tf.subtract(tensor1, tensor2) : 두 개의 텐서사이의 빼기 연산을 수행.
#   - tf.square(tensor) : 각각의 텐서 값에 제곱을 수행.
#   - tf.reduce_sum(tensor, dimension) : 텐서의 dimension 차원에 해당하는 값을 더하고 차원을 축소. 여기서는 x, y 값을 더해주는 역할
#   - tf.argmin(tensor, dimension) : dimension(0:열, 1:행, 2:면...), 0 이면 열을 기준으로 가장 작은 값을 가지는 행의 index 값을 출력.

assignments = tf.argmin(tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2), 0)

#  3단계 : 각 그룹에 대해 새로운 중심을 계산
#   - tf.equal(tensor, cluster) : tensor 값에 대해 cluster 와 비교해서 같으면 True, 다르면 False 를 출력.
#   - tf.where(condition) : condition 을 만족하는 값에 대해 해당 인덱스를 출력.
#   - tf.reshape(tensor, shape) : tensor 의 차원을 shape 에 맞게 변형.
#   - tf.gather(tensor, indices) : tensor 에 대해 indices 를 만족하는 값을 출력.
#   - tf.reduce_mean(tensor, axis) : tensor 를 axis 에 대해 평균 값을 구해서 출력.
clustered_data = [tf.gather(vectors, tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1])) for c in range(k)]
means = tf.concat([tf.reduce_mean(data, axis=1) for data in clustered_data], 0)
# means = tf.concat([tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1])), axis=1) for c in range(k)], 0)

#   - tf.assign(tensor1, tensor2) : tensor1 을 tensor2 로 update 수행.
update_centroids = tf.assign(centroids, means)

#   - tf.global_variables_initializer() : 사용된 텐서 변수들에 대해 초기화.
init_op = tf.global_variables_initializer()

#   - tf.Session() : 텐서플로우 세션 생성, sess.run(init_op) : 텐서 변수 초기화 수행.
sess = tf.Session()
sess.run(init_op)

for step in range(100):
    _, centroid_values, assignment_values = sess.run([update_centroids, centroids, assignments])

colors = ["g.", "r.", "c.", "y."]

#  군집별로 점 출력
for i in range(k):
    temp_data = sess.run(clustered_data[i])
    plt.plot(temp_data[0, :, 0], temp_data[0, :, 1], colors[i], markersize=10)

#  중심값 출력
plt.scatter(centroid_values[:, 0], centroid_values[:, 1], marker='x', s=150, linewidths=5, zorder=10)
plt.show()




###################################################################################################
## 영어 수학 데이터
###################################################################################################
## ▣ K-평균 알고리즘
##  - 주어진 데이터를 지정된 군집 개수(K)로 그룹화 한다.
##    한 군집 내의 데이터들은 동일한 성질을 가지며 다른 그룹과는 구별된다.
##    알고리즘의 결과는 중심이라고 부르는 K개의 점으로서, 이들은 각기 다른 그룹의 중심점을 나타내며 데이터들은
##    K개의 군집 중 하나에만 속할 수 있다.
##    한 군집 내의 모든 데이터들은 다른 어떤 중심들보다 자기 군집 중심과의 거리가 더 가깝다.
##  - 반복 개선 기법 사용
##   1. 초기 단계(0 단계) : K개 중심의 초기 집합을 결정 (K 개를 임의로 선택)
##   2. 할당 단계(1 단계) : 각 데이터를 가장 가까운 군집에 할당
##   3. 업데이트 단계(2 단계) : 각 그룹에 대해 새로운 중심을 계산
###################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

data= np.loadtxt("D://data//academy.csv",delimiter=',', skiprows=1)
x_data = list(data[:,-3])
y_data = list(data[:,-4])


num_points = 2000
vectors_set = []

for i in range(num_points):
    if np.random.random() > 0.5:
        vectors_set.append([np.random.normal(0.0, 0.9), np.random.normal(0.0, 0.9)])
    else:
        vectors_set.append([np.random.normal(3.0, 0.5), np.random.normal(1.0, 0.5)])

# df = pd.DataFrame({'x': [v[0] for v in vectors_set],
#                    'y': [v[1] for v in vectors_set]})
x = [v[0] for v in vectors_set]
y = [v[1] for v in vectors_set]
# plt.plot(x, y, 'ro')
# plt.show()

vectors = tf.constant(vectors_set)
k = 4

#  1단계 : 랜덤으로 K 개의 초기 집합을 설정
#   - tf.random_shuffle(object) : 상수 텐서를 랜덤으로 섞는다.
#   - tf.slice(object, begin, size) : 랜덤으로 얻어진 상수 텐서를 처음부터 k개 만큼 자른다.
#   - tf_Variable(object) : 상수 텐서를 변수 텐서로 변환.
centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0, 0], [k, -1]))

#   - expanded_vectors   : (2000, 2) --> (1, 2000, 2), expanded_vectors.shape
#   - expanded_centroids : (4, 2) --> (4, 1, 2), expanded_centroids.shape
#   ※ 크기가 1인 차원만 broadcasting 기능이 동작하므로, 두 텐서간의 차원 확장이 필요하다.
expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroids = tf.expand_dims(centroids, 1)

#  2단계 : 각 데이터를 가장 가까운 군집에 할당
#  → 중심과 각 점간의 거리를 구하기 위해 유클리드 제곱 거리 알고리즘을 선택.
#   - tf.subtract(tensor1, tensor2) : 두 개의 텐서사이의 빼기 연산을 수행.
#   - tf.square(tensor) : 각각의 텐서 값에 제곱을 수행.
#   - tf.reduce_sum(tensor, dimension) : 텐서의 dimension 차원에 해당하는 값을 더하고 차원을 축소. 여기서는 x, y 값을 더해주는 역할
#   - tf.argmin(tensor, dimension) : dimension(0:열, 1:행, 2:면...), 0 이면 열을 기준으로 가장 작은 값을 가지는 행의 index 값을 출력.

assignments = tf.argmin(tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2), 0)

#  3단계 : 각 그룹에 대해 새로운 중심을 계산
#   - tf.equal(tensor, cluster) : tensor 값에 대해 cluster 와 비교해서 같으면 True, 다르면 False 를 출력.
#   - tf.where(condition) : condition 을 만족하는 값에 대해 해당 인덱스를 출력.
#   - tf.reshape(tensor, shape) : tensor 의 차원을 shape 에 맞게 변형.
#   - tf.gather(tensor, indices) : tensor 에 대해 indices 를 만족하는 값을 출력.
#   - tf.reduce_mean(tensor, axis) : tensor 를 axis 에 대해 평균 값을 구해서 출력.
clustered_data = [tf.gather(vectors, tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1])) for c in range(k)]
means = tf.concat([tf.reduce_mean(data, axis=1) for data in clustered_data], 0)
# means = tf.concat([tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1])), axis=1) for c in range(k)], 0)

#   - tf.assign(tensor1, tensor2) : tensor1 을 tensor2 로 update 수행.
update_centroids = tf.assign(centroids, means)

#   - tf.global_variables_initializer() : 사용된 텐서 변수들에 대해 초기화.
init_op = tf.global_variables_initializer()

#   - tf.Session() : 텐서플로우 세션 생성, sess.run(init_op) : 텐서 변수 초기화 수행.
sess = tf.Session()
sess.run(init_op)

for step in range(100):
    _, centroid_values, assignment_values = sess.run([update_centroids, centroids, assignments])

colors = ["g.", "r.", "c.", "y."]

#  군집별로 점 출력
for i in range(k):
    temp_data = sess.run(clustered_data[i])
    plt.plot(temp_data[0, :, 0], temp_data[0, :, 1], colors[i], markersize=10)

#  중심값 출력
plt.scatter(centroid_values[:, 0], centroid_values[:, 1], marker='x', s=150, linewidths=5, zorder=10)
plt.show()




