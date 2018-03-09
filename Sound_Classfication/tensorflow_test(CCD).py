import glob
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import tensorflow.contrib as con
# sound_names = ["air conditioner","car horn","children playing","dog bark","drilling","engine idling",
#                "gun shot","jackhammer","siren","street music"]

# sound_data = np.load('D:\park\ccd_sound_data\\ccd_sound.npz')
sound_data = np.load('D:\park\\urban_sound.npz')
X_data = sound_data['X']
y_data = sound_data['y']

sound_groups = sound_data['groups']


# print(X_data[0])
# print(y_data[33])
#
#
# print(X_data.shape)
# print(y_data.shape)

X_sub, X_test, y_sub, y_test = train_test_split(X_data, y_data, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_sub, y_sub, test_size=0.2)

# print(len(X_train), len(X_val), len(X_test), len(X_sub))
# print(len(y_train), len(y_val), len(y_test), len(y_sub))
# print(X_data.shape, y_data.shape)

training_epochs = 500
n_dim = 193
n_classes = 10
n_hidden_units_one = 300
n_hidden_units_two = 200
n_hidden_units_three = 100
learning_rate = 0.01
sd = 1 / np.sqrt(n_dim)

sd1 = 1 / np.sqrt(n_hidden_units_one)
sd2 = 1 / np.sqrt(n_hidden_units_two)
sd3 = 1 / np.sqrt(n_hidden_units_three)
sd4 = 1 / np.sqrt(n_classes)




X = tf.placeholder(tf.float32, [None, n_dim])
Y = tf.placeholder(tf.float32, [None, n_classes])


def parametric_relu(_x, name):
    alphas = tf.get_variable(name, _x.get_shape()[-1], initializer=tf.constant_initializer(0.01), dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5
    return pos + neg

# W_1 = tf.Variable(tf.random_normal([n_dim, n_hidden_units_one], mean=0, stddev=sd), name='W1')
W_1 = tf.get_variable(name='W1', shape=[n_dim, n_hidden_units_one], initializer=tf.contrib.layers.variance_scaling_initializer())
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean=0, stddev=sd), name="b1")
h_1 = tf.nn.sigmoid(tf.matmul(X, W_1) + b_1)
# h_1 = tf.nn.relu(tf.matmul(X, W_1) + b_1)
# h_1 = parametric_relu(tf.matmul(X, W_1) + b_1, 'h1')

# W_2 = tf.Variable(tf.random_normal([n_hidden_units_one, n_hidden_units_two], mean=0, stddev=sd), name="W2")
W_2 = tf.get_variable(name='W2', shape=[n_hidden_units_one, n_hidden_units_two], initializer=tf.contrib.layers.variance_scaling_initializer())
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean=0, stddev=sd), name="b2")
h_2 = tf.nn.tanh(tf.matmul(h_1, W_2) + b_2)
# h_2 = tf.nn.relu(tf.matmul(h_1, W_2) + b_2)

# W_3 = tf.Variable(tf.random_normal([n_hidden_units_two, n_hidden_units_three], mean=0, stddev=sd), name="W3")
W_3 = tf.get_variable(name='W3', shape=[n_hidden_units_two, n_hidden_units_three], initializer=tf.contrib.layers.variance_scaling_initializer())
b_3 = tf.Variable(tf.random_normal([n_hidden_units_three], mean=0, stddev=sd), name="b3")
h_3 = tf.nn.sigmoid(tf.matmul(h_2, W_3) + b_3)
# h_3 = tf.nn.relu(tf.matmul(h_2, W_3) + b_3)
# h_3 = parametric_relu(tf.matmul(h_2, W_3) + b_3, 'h3')

# W = tf.Variable(tf.random_normal([n_hidden_units_three, n_classes], mean=0, stddev=sd), name="W")
W = tf.get_variable(name='W', shape=[n_hidden_units_three, n_classes], initializer=tf.contrib.layers.variance_scaling_initializer())
b = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=sd), name="b")
y_ = tf.nn.softmax(tf.matmul(h_3, W) + b)

# cost_funtion = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_), reduction_indices=[1])) + (0.01/(2*tf.to_float(tf.shape(Y)[0])))*tf.reduce_sum(tf.square(W))
cost_funtion = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_funtion)
# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_funtion)

correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()
cost_history = np.empty(shape=[1], dtype=float)

confusion_mat = np.zeros((10,10))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    stime = time.time()
    sess.run(init)
    for epoch in range(0, training_epochs+1):
        sstime = time.time()
        _, cost = sess.run([optimizer, cost_funtion], feed_dict={X: X_sub, Y: y_sub})
        if epoch%1000 == 0:
            print(epoch, sess.run([cost_funtion], feed_dict={X: X_sub, Y: y_sub}))

        cost_history = np.append(cost_history, cost)
    print("Test Accuracy : ", round(sess.run(accuracy, feed_dict={X: X_test, Y: y_test}), 3))
    print("Train Accuracy : ", round(sess.run(accuracy, feed_dict={X: X_train, Y: y_train}), 3))
    print("validation Accuracy : ", round(sess.run(accuracy, feed_dict={X: X_val, Y: y_val}), 3))
    etime = time.time()
    print('consumption time : ', round(etime-stime, 6))
    saver.save(sess, "D:\park\\20180309\\model_A(0.01)_he\\model_A(0.01)_he.ckpt")
    ensemble_confusion_mat = con.metrics.confusion_matrix(labels=tf.argmax(y_, 1), predictions=tf.argmax(Y, 1),
                                                                        num_classes=10)
    print(sess.run(ensemble_confusion_mat))



fig = plt.figure(figsize=(10,8))
plt.plot(cost_history)
plt.ylabel("Cost")
plt.xlabel("Epoch")
plt.axis([0,training_epochs,0,np.max(cost_history)])
plt.show()
