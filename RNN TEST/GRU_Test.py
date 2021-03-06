import os
import numpy as np
import tensorflow as tf
import time
import re

class Model:
    def __init__(self, sess, n_inputs, n_sequences, n_hiddens, n_outputs, hidden_layer_cnt, file_name, model_name):
        self.sess = sess
        self.n_inputs = n_inputs
        self.n_sequences = n_sequences
        self.n_hiddens = n_hiddens
        self.n_outputs = n_outputs
        self.hidden_layer_cnt = hidden_layer_cnt
        self.file_name = file_name
        self.model_name = model_name
        self.regularizer = tf.contrib.layers.l2_regularizer(0.001)
        self.training = True
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.model_name):
            self.learning_rate = 0.001

            self.X = tf.placeholder(tf.float32, [None, self.n_sequences, self.n_inputs])
            self.Y = tf.placeholder(tf.float32, [None, self.n_outputs])

            self.multi_cells = tf.contrib.rnn.MultiRNNCell([self.gru_cell(self.n_hiddens) for _ in range(self.hidden_layer_cnt)], state_is_tuple=False)
            # self.multi_cells = tf.contrib.rnn.DropoutWrapper(self.multi_cells, output_keep_prob=0.5)
            self.outputs, _states = tf.nn.dynamic_rnn(self.multi_cells, self.X, dtype=tf.float32)
            self.fc_1 = tf.contrib.layers.fully_connected(self.outputs[:, -1], 250, activation_fn=None)
            self.Y_ = tf.contrib.layers.fully_connected(self.fc_1, self.n_outputs, activation_fn=None)

            self.reg_loss = tf.reduce_sum([self.regularizer(train_var) for train_var in tf.trainable_variables() if re.search('(kernel)|(weights)', train_var.name) is not None])
            self.loss = tf.reduce_sum(tf.square(self.Y_ - self.Y)) + self.reg_loss
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

            self.targets = tf.placeholder(tf.float32, [None, 1])
            self.predictions = tf.placeholder(tf.float32, [None, 1])
            self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.targets - self.predictions)))


    def lstm_batch_norm(self, inputs, shape, is_training, epsilon=1e-3, decay=0.99):
        scale = tf.Variable(tf.constant(1.0, shape=[shape]), trainable=True)
        beta = tf.Variable(tf.constant(0.0, shape=[shape]), trainable=True)

        population_mean = tf.get_variable('population_mean', [shape], trainable=False)
        population_var = tf.get_variable('population_var', [shape], trainable=False)
        batch_mean, batch_var = tf.nn.moments(inputs, [0])

        train_mean_op = tf.assign(population_mean, population_mean * decay + batch_mean * (1 - decay))
        train_var_op = tf.assign(population_var, population_var * decay + batch_var * (1 - decay))

        if is_training is True:
            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)
        else:
            return tf.nn.batch_normalization(inputs, population_mean, population_var, beta, scale, epsilon)


    def gru_cell(self, hidden_size):
       cell = tf.nn.rnn_cell.GRUCell(hidden_size, activation=None)
       cell = self.lstm_batch_norm(cell, hidden_size, self.training, epsilon=1e-3, decay=0.99)
       cell = tf.tanh(cell)
       if self.training:
           cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=0.5)
       return cell

    # def gru_cell(self, hidden_size):
    #     cell = tf.nn.rnn_cell.GRUCell(hidden_size, activation=tf.tanh)
    #     if self.training is True:
    #         cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=0.5)
    #     return cell

    def train(self, x_data, y_data):
        self.training = True
        return self.sess.run([self.reg_loss, self.loss, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data})

    def predict(self, x_data):
        self.training = False
        return self.sess.run(self.Y_, feed_dict={self.X: x_data})

    def rmse_predict(self, targets, predictions):
        self.training = False
        return self.sess.run(self.rmse, feed_dict={self.targets: targets, self.predictions: predictions})

n_inputs = 7
n_sequences = 5
n_hiddens = 2
n_outputs = 1
hidden_layer_cnt = 5

def min_max_scaler(data):
    return (data - np.min(data, axis=0))/(np.max(data, axis=0) - np.min(data, axis=0) + 1e-5)

def read_data(file_name):
    data = np.loadtxt('D:\\bitcoin/'+file_name, delimiter=',', skiprows=1)
    data = data[:, 1:]
    data = data[np.sum(np.isnan(data), axis=1) == 0]
    data = min_max_scaler(data)
    x, y = data, data[:, [3]]
    dataX = []
    dataY = []
    for i in range(0, len(data) - n_sequences):
        _x = x[i:i + n_sequences]
        _y = y[i + n_sequences]
        dataX.append(_x)
        dataY.append(_y)
    return dataX, dataY

file_list = os.listdir('D:\\bitcoin')
model_list = []

batch_size = 100
epochs = 20






# def lstm_batch_norm(inputs, name_scope, is_training, epsilon=1e-3, decay=0.99):
#     with tf.variable_scope(name_scope):
#         size = inputs.get_shape().as_list()[1]
#
#         scale = tf.get_variable('scale', [size], initializer=tf.constant_initializer(0.1))
#         offset = tf.get_variable('offset', [size])
#
#         population_mean = tf.get_variable('population_mean', [size], initializer=tf.zeros_initializer, trainable=False)
#         population_var = tf.get_variable('population_var', [size], initializer=tf.ones_initializer, trainable=False)
#         batch_mean, batch_var = tf.nn.moments(inputs, [0])
#
#
#         train_mean_op = tf.assign(population_mean, population_mean * decay + batch_mean * (1 - decay))
#         train_var_op = tf.assign(population_var, population_var * decay + batch_var * (1 - decay))
#
#         if is_training is True:
#             with tf.control_dependencies([train_mean_op, train_var_op]):
#                 return tf.nn.batch_normalization(inputs, batch_mean, batch_var, offset, scale, epsilon)
#         else:
#             return tf.nn.batch_normalization(inputs, population_mean, population_var, offset, scale, epsilon)





with tf.Session() as sess:
    for idx, file_name in enumerate(file_list):
        model_list.append(Model(sess=sess, n_inputs=n_inputs, n_sequences=n_sequences, n_hiddens=n_hiddens,
                                n_outputs=n_outputs, hidden_layer_cnt=hidden_layer_cnt, file_name=file_name, model_name='Model_'+str(idx+1)))

    sess.run(tf.global_variables_initializer())

    for model in model_list:
        total_X, total_Y = read_data(model.file_name)  # 모델별 파일 로딩
        train_X, train_Y = total_X[:int(len(total_Y)*0.7)], total_Y[:int(len(total_Y)*0.7)]  # train 데이터
        test_X, test_Y = total_X[int(len(total_Y)*0.7):], total_Y[int(len(total_Y)*0.7):]  # test 데이터
        train_len, test_len = len(train_Y), len(test_Y)

        stime = time.time()
        print(model.model_name, ', training start -')
        print('train data -', train_len, ', test data -', test_len)
        for epoch in range(epochs):
            train_loss = 0.
            for idx in range(0, train_len, batch_size):
                sample_size = train_len if batch_size > train_len else batch_size
                batch_X, batch_Y = train_X[idx: idx+sample_size], train_Y[idx: idx+sample_size]
                reg_loss, loss, _ = model.train(batch_X, batch_Y)
                train_loss += loss / sample_size
                train_len -= sample_size
            print('Model :', model.model_name, ', epoch :', epoch+1, ', loss :', train_loss)
            train_len, test_len = len(train_Y), len(test_Y)
        print(model.model_name, ', training end -\n')

        print(model.model_name, ', testing start -')
        test_rmse = 0.
        for idx in range(0, test_len, batch_size):
            sample_size = test_len if batch_size > test_len else batch_size
            batch_X, batch_Y = test_X[idx: idx + sample_size], test_Y[idx: idx + sample_size]
            predicts = model.predict(batch_X)
            rmse = model.rmse_predict(batch_Y, predicts)
            test_rmse += rmse / sample_size
            test_len -= sample_size
        etime = time.time()
        print('Model :', model.model_name, ', rmse :', test_rmse)
        print(model.model_name, ', testing end -')
        print(model.model_name, ', time -', etime-stime, '\n')