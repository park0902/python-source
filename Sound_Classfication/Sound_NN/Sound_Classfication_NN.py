import glob
import librosa
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

# Sound Names
sound_names = ["air conditioner","car horn","children playing","dog bark","drilling","engine idling",
               "gun shot","jackhammer","siren","street music"]

# 파일 로드(5가지 Sound 음성 패턴 추출한.npz 파일)
sound_data = np.load('D:\\park\\urban_sound.npz')

# 데이터 추출(X_data : 음성 추출 데이터, y_data : 라벨(0~9))
X_data = sound_data['X']
y_data = sound_data['y']

# sound_groups = sound_data['groups']

# print(X_data[0])
# print(y_data[0])
# print(sound_groups[0])

# print(X_data.shape)
# print(y_data.shape)
# print(sound_groups.shape)

# 트레이닝 셋 분할(Training Set 60%, Test Set 20%, Validation Set 20%)
X_sub, X_test, y_sub, y_test = train_test_split(X_data, y_data, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_sub, y_sub, test_size=0.2)

# print(len(X_train), len(X_val), len(X_test), len(X_sub))
# print(len(y_train), len(y_val), len(y_test), len(y_sub))
# print(X_data.shape, y_data.shape)

# 모델링 변수
training_epochs = 1000
n_dim = 193
n_classes = 10
n_hidden_units_one = 300
n_hidden_units_two = 200
n_hidden_units_three = 100
learning_rate = 0.1
sd = 1 / np.sqrt(n_dim)
confusion_mat = np.zeros((10, 10))


# 입/출력 레이어 설정
X = tf.placeholder(tf.float32, [None, n_dim])
Y = tf.placeholder(tf.float32, [None, n_classes])

# 첫번째 히든 레이어
W_1 = tf.Variable(tf.random_normal([n_dim, n_hidden_units_one], mean=0, stddev=sd), name='W1')
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean=0, stddev=sd), name="b1")
h_1 = tf.nn.sigmoid(tf.matmul(X, W_1) + b_1)

# 두번째 히든 레이어
W_2 = tf.Variable(tf.random_normal([n_hidden_units_one, n_hidden_units_two], mean=0, stddev=sd), name="W2")
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean=0, stddev=sd), name="b2")
h_2 = tf.nn.tanh(tf.matmul(h_1, W_2) + b_2)

# 세번째 히든 레이어
W_3 = tf.Variable(tf.random_normal([n_hidden_units_two, n_hidden_units_three], mean=0, stddev=sd), name="W3")
b_3 = tf.Variable(tf.random_normal([n_hidden_units_three], mean=0, stddev=sd), name="b3")
h_3 = tf.nn.sigmoid(tf.matmul(h_2, W_3) + b_3)

# 출력 레이어
W = tf.Variable(tf.random_normal([n_hidden_units_three, n_classes], mean=0, stddev=sd), name="W")
b = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=sd), name="b")
y_ = tf.nn.softmax(tf.matmul(h_3, W) + b)

# 비용함수
cost_funtion = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_), reduction_indices=[1]))

# 최적화 함수
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_funtion)

# 정확도 예측
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 비용함수 값 저장
cost_history = np.empty(shape=[1], dtype=float)

# Tensorflow 모델(.ckpt) 파일 저장하기 위한 설정
saver = tf.train.Saver()

# 모든 변수 초기화
init = tf.global_variables_initializer()

# Tensorflow 학습 시작 및 정확도 계산
with tf.Session() as sess:
    stime = time.time()
    sess.run(init)
    for epoch in range(0, training_epochs+1):
        sstime = time.time()
        _, cost = sess.run([optimizer, cost_funtion], feed_dict={X: X_sub, Y: y_sub})
        if epoch % 100 == 0:
            print(epoch, sess.run([cost_funtion], feed_dict={X: X_sub, Y: y_sub}))

        cost_history = np.append(cost_history, cost)
    print("Valdation Accuracy : ", round(sess.run(accuracy, feed_dict={X: X_test, Y: y_test}), 3))
    etime = time.time()
    print('consumption time : ', round(etime-stime, 6))

    # 학습된 모델(.ckpt) 저장
    saver.save(sess, "D:\\park\\Python_Project\\py-github_project\\python-source\\Sound_Data\\\model_B\\model_B.ckpt")



