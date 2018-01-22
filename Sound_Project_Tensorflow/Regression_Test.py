# import tensorflow as tf
# import matplotlib.pyplot as plt
#
# x_train = [1,2,3]
# y_train = [1,2,3]
#
# w = tf.Variable(tf.random_normal([1]), name='weight')
# b = tf.Variable(tf.random_normal([1]), name='bias')
# X = tf.placeholder(tf.float32)
# Y = tf.placeholder(tf.float32)
#
# hypothesis = x_train * w + b
#
# cost = tf.reduce_mean(tf.square(hypothesis - y_train))
#
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
#
# train = optimizer.minimize(cost)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# for step in range(2001):
#     cost_val, W_val, b_val, _ = sess.run([cost, w, b, train],
#                                          feed_dict={X: [1,2,3,4,5], Y: [2.1, 3.1, 4.1, 5.1, 6.1]})
#     if step % 20 == 0:
#         print(step, cost_val, W_val, b_val)
#
# print(sess.run(hypothesis, feed_dict={X: [5]}))
# print(sess.run(hypothesis, feed_dict={X: [2.5]}))
# print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))


import tensorflow as tf
import matplotlib.pyplot as plt

# 데이터 셋
x_data = [4.039, 1.3197, 9.5613, 0.5978, 3.5316, 0.1540, 1.6899, 7.3172, 4.5092, 2.9632]
y_data = [11.4215, 10.0112, 30.2991, 1.0625, 13.1776, -3.1976, 6.7367, 23.8550, 14.8951, 11.6137]
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 모델
W = tf.Variable(tf.random_uniform([1], -5.0, 5.0))
b = tf.Variable(tf.random_uniform([1], -5.0, 5.0))

# 가설
hypothesis = W * X + b

# 비용함수
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y)) + (0.01/(2*tf.to_float(tf.shape(Y[0]))))*tf.reduce_sum(tf.square(W))

# 경사하강법 최적화
rate = tf.Variable(0.01)
optimizer = tf.train.GradientDescentOptimizer(rate)
# optimizer = tf.train.AdamOptimizer(a)

# 비용함수 최소화
train = optimizer.minimize(cost)

# 세션 생성&초기화
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step %20 == 0:
        print(step, sess.run([train,cost], feed_dict={X:x_data, Y:y_data}), sess.run(W), sess.run(b))
        # print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W), sess.run(b))

answer = sess.run(hypothesis, feed_dict={X: 5})

print('When X=5, hypothesis = '+str(answer))

plt.figure(1)
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(x_data, y_data, 'ro')
plt.plot(x_data, sess.run(W) * x_data + sess.run(b), 'b')
plt.plot([5], answer, 'go')
plt.show()

