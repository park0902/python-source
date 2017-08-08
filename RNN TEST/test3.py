# '''
# This script shows how to predict stock prices using a basic RNN
# '''
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# import os
#
# tf.set_random_seed(777)  # reproducibility
#
#
# def MinMaxScaler(data):
#     numerator = data - np.min(data, 0)
#     denominator = np.max(data, 0) - np.min(data, 0)
#     return numerator / (denominator + 1e-7)
#
#
# # train Parameters
# seq_length = 7
# data_dim = 8
# hidden_dim = 100
# output_dim = 1
# learning_rate = 0.01
# iterations = 1000
#
# # high	    diff_24h	diff_per_24h    bid	      ask	    low	      volume        last
# # 2318.82	2228.7	    4.043612869     2319.4	  2319.99	2129.78	  4241.641516	2318.82
# xy = np.loadtxt('d:\\data\\bitcoin_okcoin_usd222.csv', delimiter=',')
# # xy = xy[::-1]  # reverse order (chronically ordered)
# xy = MinMaxScaler(xy)
# x = xy
# y = xy[:, [-1]]  # last as label
#
# # build a dataset
# dataX = []
# dataY = []
# for i in range(0, len(y) - seq_length):
#     _x = x[i:i + seq_length]
#     _y = y[i + seq_length]  # Next last price
#     # print(_x, "->", _y)
#     dataX.append(_x)
#     dataY.append(_y)
#
#
# # train/test split
# train_size = int(len(dataY) * 0.7)
# test_size = len(dataY) - train_size
# trainX, testX = np.array(dataX[0:train_size]), np.array(
#     dataX[train_size:len(dataX)])
# trainY, testY = np.array(dataY[0:train_size]), np.array(
#     dataY[train_size:len(dataY)])
#
#
# # input place holders
# X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
# Y = tf.placeholder(tf.float32, [None, 1])
#
#
# # build a LSTM network
# cell = tf.contrib.rnn.BasicLSTMCell(
#     num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
# outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
# Y_pred = tf.contrib.layers.fully_connected(
#     outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output
#
#
# # cost/loss
# loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
#
#
# # optimizer
# optimizer = tf.train.AdamOptimizer(learning_rate)
# train = optimizer.minimize(loss)
#
#
# # RMSE
# targets = tf.placeholder(tf.float32, [None, 1])
# predictions = tf.placeholder(tf.float32, [None, 1])
# rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))
#
# with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#
#     # Training step
#     for i in range(iterations):
#         _, step_loss = sess.run([train, loss], feed_dict={
#                                 X: trainX, Y: trainY})
#         print("[step: {}] loss: {}".format(i, step_loss))
#
#
#     # Test step
#     test_predict = sess.run(Y_pred, feed_dict={X: testX})
#     rmse_val = sess.run(rmse, feed_dict={
#                     targets: testY, predictions: test_predict})
#     print("RMSE: {}".format(rmse_val))
#
#
# # Plot predictions
# plt.plot(testY)
# plt.plot(test_predict)
# plt.xlabel("Time Period")
# plt.ylabel("Stock Price")
# plt.show()




import tensorflow as tf
import numpy as np
from tensorflow.contrib import *
import time

class Model:
    def __init__(self, n_inputs, n_sequences, n_hiddens, n_outputs, hidden_layer_cnt, file_name, model_name):
        self.n_inputs = n_inputs
        self.n_sequences = n_sequences
        self.n_hiddens = n_hiddens
        self.n_outputs = n_outputs
        self.hidden_layer_cnt = hidden_layer_cnt
        self.file_name = file_name
        self.model_name = model_name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.model_name):
            with tf.name_scope('input_layer'):
                self.X = tf.placeholder(tf.float32, [None, self.n_sequences, self.n_inputs])
                self.Y = tf.placeholder(tf.float32, [None, self.n_outputs])

            with tf.name_scope('GRU'):
                self.cell = tf.nn.rnn_cell.GRUCell(num_units=self.n_hiddens, activation=tf.tanh)
                self.cell = tf.nn.rnn_cell.DropoutWrapper(self.cell, output_keep_prob=0.5)
                self.multi_cell = tf.nn.rnn_cell.MultiRNNCell([self.cell] * self.hidden_layer_cnt, state_is_tuple= False)
                outputs, _states = tf.nn.dynamic_rnn(self.multi_cell, self.X, dtype=tf.float32)
                self.Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], self.n_outputs, activation_fn=None)

            # with tf.name_scope('GRU'):
            #     self.cell = tf.nn.rnn_cell.GRUCell(num_units=self.n_hiddens, activation=tf.tanh)
            #     self.cell = tf.nn.rnn_cell.DropoutWrapper(self.cell, output_keep_prob=0.5)
            #     self.multi_cell = tf.nn.rnn_cell.MultiRNNCell([self.cell] * self.hidden_layer_cnt)
            #     outputs, _states = tf.nn.dynamic_rnn(self.multi_cell, self.X, dtype=tf.float32)
            #     self.FC = tf.reshape(outputs, [-1, self.n_hiddens])
            #     self.outputs = tf.contrib.layers.fully_connected(self.FC, self.n_outputs, activation_fn=None)
            #     self.outputs = tf.reshape(self.n_outputs, [None, self.n_sequences, self.n_outputs])
            #
            #     self.sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=self.outputs, targets=self.Y)
            #     self.loss = tf.reduce_mean(self.sequence_loss)

            # with tf.name_scope('LSTM'):
            #     self.cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.n_hiddens, state_is_tuple=True, activation=tf.tanh)
            #     self.cell = tf.nn.rnn_cell.DropoutWrapper(self.cell, input_keep_prob=0.5)
            #     self.multi_cell = tf.nn.rnn_cell.MultiRNNCell([self.cell] * self.hidden_layer_cnt)
            #     self.multi_cell = tf.nn.rnn_cell.DropoutWrapper(self.multi_cell, output_keep_prob=0.5)
            #     outputs, _states = tf.nn.dynamic_rnn(self.multi_cell, self.X, dtype=tf.float32)
            #     self.Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], self.n_outputs, activation_fn=None)


        # self.flat = tf.reshape(outputs, [-1, self.n_hiddens])
        # self.logits = layers.linear(self.flat, self.n_inputs)
        #
        #
        # self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost)
        # self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits, 1), tf.arg_max(self.Y, 1)), dtype=tf.float32))
        # # self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.Y, tf.cast(self.Y, tf.uint8)), dtype=tf.float32))


        self.loss = tf.reduce_sum(tf.square(self.Y_pred-self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        self.train = self.optimizer.minimize(self.loss)


def min_max_scaler(data):
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0) + 1e-5)


def read_data(file_name):
    data = np.loadtxt('d:/data/' + file_name, delimiter=',', skiprows=1)
    data = data[:, 1:]
    data = data[np.sum(np.isnan(data), axis=1) == 0]
    data = min_max_scaler(data)
    # return data, data[:, [3]]
    return data


def data_setting(data, sequence_length):
    dataX = []
    dataY = []
    x = data
    y = data[:, [-3]]
    for i in range(0, len(y)-sequence_length):
        x = x[i:i+sequence_length]
        y = y[i:i+sequence_length]
        dataX.append(x)
        dataY.append(y)
    train_size = int(len(dataY)*0.7)
    # test_size = len(dataY) - train_size
    trainX, trainY = np.array(dataX[0:train_size]), np.array(dataY[0:train_size])
    testX, testY = np.array(dataX[train_size:len(dataX)]), np.array(dataY[train_size:len(dataY)])
    return trainX, trainY, testX, testY



# print(read_data('bitstampUSD_1-min_data_2012-01-01_to_2017-05-31.csv'))

file_names=['bitstampUSD_1-min_data_2012-01-01_to_2017-05-31.csv']

batch_size = 1000
n_inputs = 7
n_sequences = 60
n_hiddens = 100
n_outputs = 1
hidden_layer_cnt = 5
iterations = 1000

model = Model(batch_size,n_sequences,n_hiddens,n_outputs,hidden_layer_cnt,file_name=file_names[0],model_name='RNN')
data = read_data(file_names[0])

trainX, trainY, testX, testY = data_setting(data, n_sequences)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={
                                X: trainX, Y: trainY})
        print("[step: {}] loss: {}".format(i, step_loss))


    # Test step
    test_predict = sess.run(Y_pred, feed_dict={X: testX})
    rmse_val = sess.run(rmse, feed_dict={
                    targets: testY, predictions: test_predict})

    print("RMSE: {}".format(rmse_val))
