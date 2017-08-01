from __future__ import division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
########################################################################################################################
########################################################################################################################
##### Manual RNN #####
# def reset_graph(seed=42):
#     tf.reset_default_graph()
#     tf.set_random_seed(seed)
#     np.random.seed(seed)
#
# plt.rcParams['axes.labelsize'] = 14
# plt.rcParams['xtick.labelsize'] = 12
# plt.rcParams['ytick.labelsize'] = 12
#
# PROJECT_ROOT_DIR = "."
# CHAPTER_ID = 'rnn'
#
# def save_fig(fig_id, tight_layout=True):
#     path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
#     print("Saving Figure", fig_id)
#     if tight_layout:
#         plt.tight_layout()
#     plt.savefig(path, format='png', dpi=300)
#
# reset_graph()
#
# n_inputs = 3
# n_neurons = 5
#
# X0 = tf.placeholder(tf.float32, [None, n_inputs])
# X1 = tf.placeholder(tf.float32, [None, n_inputs])
#
# Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons], dtype=tf.float32))
# Wy = tf.Variable(tf.random_normal(shape=[n_neurons, n_neurons], dtype=tf.float32))
# b = tf.Variable(tf.zeros([1, n_neurons], dtype=tf.float32))
#
# Y0 = tf.tanh(tf.matmul(X0,Wx) + b)
# Y1 = tf.tanh(tf.matmul(Y0,Wy) + tf.matmul(X1, Wx) + b)
#
# init = tf.global_variables_initializer()
#
# X0_batch = np.array([[0,1,2], [3,4,5], [6,7,8], [9,0,1]])
# X1_batch = np.array([[9,8,7], [0,0,0], [6,5,4], [3,2,1]])
#
# with tf.Session() as sess:
#     init.run()
#     Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1: X1_batch})
#
# print(Y0_val)
# print(Y1_val)
########################################################################################################################



########################################################################################################################
##### Using static_rnn() #####
# def reset_graph(seed=42):
#     tf.reset_default_graph()
#     tf.set_random_seed(seed)
#     np.random.seed(seed)
#
# n_inputs = 3
# n_neurons = 5
#
# reset_graph()
#
# X0 = tf.placeholder(tf.float32, [None, n_inputs])
# X1 = tf.placeholder(tf.float32, [None, n_inputs])
#
# basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
# output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, [X0, X1], dtype=tf.float32)
#
# Y0, Y1 = output_seqs
#
# init = tf.global_variables_initializer()
#
# X0_batch = np.array([[0,1,2], [3,4,5], [6,7,8], [9,0,1]])
# X1_batch = np.array([[9,8,7], [0,0,0], [6,5,4], [3,2,1]])
#
# with tf.Session() as sess:
#     init.run()
#     Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1: X1_batch})
#
# print(Y0_val)
# print(Y1_val)
########################################################################################################################



########################################################################################################################
##### Packing Sequences #####
# def reset_graph(seed=42):
#     tf.reset_default_graph()
#     tf.set_random_seed(seed)
#     np.random.seed(seed)
#
# n_steps = 2
# n_inputs = 3
# n_neurons = 5
#
# reset_graph()
#
# X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
# X_seqs = tf.unstack(tf.transpose(X, perm=[1,0,2]))
#
# basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
# output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, X_seqs, dtype=tf.float32)
#
# outputs = tf.transpose(tf.stack(output_seqs), perm=[1,0,2])
#
# init = tf.global_variables_initializer()
#
# X_batch = np.array([
#     #  t = 0       t = 1
#     [[0, 1, 2], [9, 8, 7]]
# ])
########################################################################################################################
char_rdic = ['h', 'e', 'l', 'o']  # id -> char
char_dic = {w: i for i, w in enumerate(char_rdic)}  # char -> id
print(char_dic)

ground_truth = [char_dic[c] for c in 'hello']
print(ground_truth)

x_data = np.array([[1, 0, 0, 0],  # h
                   [0, 1, 0, 0],  # e
                   [0, 0, 1, 0],  # l
                   [0, 0, 1, 0]],  # l
                  dtype='f')

x_data = tf.one_hot(ground_truth[:-1], len(char_dic), 1.0, 0.0, -1)
print(x_data)

# Configuration
rnn_size = len(char_dic)  # 4
batch_size = 1
output_size = 4

# RNN Model
rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=rnn_size,
                                       input_size=None,  # deprecated at tensorflow 0.9
                                       # activation = tanh,
                                       )

initial_state = rnn_cell.zero_state(batch_size, tf.float32)
initial_state_1 = tf.zeros([batch_size, rnn_cell.state_size])  # 위 코드와 같은 결과

x_split = tf.split(x_data, len(char_dic), 0)  # 가로축으로 4개로 split

outputs, state = tf.contrib.rnn.static_rnn(rnn_cell, x_split, initial_state)

print(outputs)
print(state)

logits = tf.reshape(tf.concat(outputs, 1),  # shape = 1 x 16
                    [-1, rnn_size])  # shape = 4 x 4
logits.get_shape()

targets = tf.reshape(ground_truth[1:], [-1])  # a shape of [-1] flattens into 1-D
targets.get_shape()

weights = tf.ones([len(char_dic) * batch_size])

loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets], [weights])
cost = tf.reduce_sum(loss) / batch_size
train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)

# Launch the graph in a session
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for i in range(100):
        sess.run(train_op)
        result = sess.run(tf.argmax(logits, 1))
        print(result, [char_rdic[t] for t in result])
