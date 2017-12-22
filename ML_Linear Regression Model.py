############################################################################
# import tensorflow as tf
# import time
#
# W = tf.Variable([.3], tf.float32)
# b = tf.Variable([-.3], tf.float32)
#
# x = tf.placeholder(tf.float32)
# y = tf.placeholder(tf.float32)
#
# linear_model = x * W + b
#
# loss = tf.reduce_sum(tf.square(linear_model - y))
#
# optimizer = tf.train.GradientDescentOptimizer(0.01)
# train = optimizer.minimize(loss)
#
# x_train = [1,2,3,4]
# y_train = [0,-1,-2,-3]
#
# stime = time.time()
#
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
#
# for i in range(1000):
#     sess.run(train, {x:x_train, y:y_train})
#
# curr_W, curr_b, curr_loss = sess.run([W,b,loss], {x:x_train, y:y_train})
#
# eetime = time.time()
#
# print("W: %s b: %s loss: %s" %(curr_W, curr_b, curr_loss))
# print('consumption time : ', round(eetime-stime, 2))
##############################################################################
import numpy as np

num_points = 1000
vectors_set = []
for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])

print(vectors_set)

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

