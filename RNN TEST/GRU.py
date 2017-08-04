# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
#
# # set random seed for comparing the two result calculations
# tf.set_random_seed(1)
#
# # this is data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#
# # hyperparameters
# lr = 0.001
# training_iters = 100000
# batch_size = 128
#
# n_inputs = 28  # MNIST data input (img shape: 28*28)
# n_steps = 28  # time steps
# n_hidden_units = 128  # neurons in hidden layer
# n_classes = 10  # MNIST classes (0-9 digits)
# num_layers = 2
#
# # tf Graph input
# x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
# y = tf.placeholder(tf.float32, [None, n_classes])
#
# # Define weights
# weights = {
#     # (28, 128)
#     'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
#     # (128, 10)
#     'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
# }
# biases = {
#     # (128, )
#     'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
#     # (10, )
#     'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
# }
# print("parameters ready")
#
#
# def RNN(X, weights, biases):
#     # hidden layer for input to cell
#     ########################################
#
#     # transpose the inputs shape from
#     # X ==> (128 batch * 28 steps, 28 inputs)
#     X = tf.reshape(X, [-1, n_inputs])
#
#     # into hidden
#     # X_in = (128 batch * 28 steps, 128 hidden)
#     X_in = tf.matmul(X, weights['in']) + biases['in']
#     # X_in ==> (128 batch, 28 steps, 128 hidden)
#     X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
#
#     # cell
#     ##########################################
#
#     # basic LSTM Cell.
#     if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
#         cell = tf.nn.rnn_cell.GRUCell(num_units=n_hidden_units)
#
#         cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)
#         cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)
#
#     else:
#         cell = tf.contrib.rnn.GRUCell(n_hidden_units)
#
#         cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.5)
#         cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)
#
#         # lstm cell is divided into two parts (c_state, h_state)
#     init_state = cell.zero_state(batch_size, dtype=tf.float32)
#
#     # You have 2 options for following step.
#     # 1: tf.nn.rnn(cell, inputs);
#     # 2: tf.nn.dynamic_rnn(cell, inputs).
#     # If use option 1, you have to modified the shape of X_in, go and check out this:
#     # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
#     # In here, we go for option 2.
#     # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
#     # Make sure the time_major is changed accordingly.
#     outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
#
#     # hidden layer for output as the final results
#     #############################################
#     # results = tf.matmul(final_state[1], weights['out']) + biases['out']
#
#     # # or
#     # unpack to list [(batch, outputs)..] * steps
#     if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
#         outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))  # states is the last outputs
#     else:
#         outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
#     results = tf.matmul(outputs[-1], weights['out']) + biases['out']  # shape = (128, 10)
#
#     return results
#
#
# pred = RNN(x, weights, biases)
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# train_op = tf.train.AdamOptimizer(lr).minimize(cost)
#
# correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#
# print("Network ready")
#
# with tf.Session() as sess:
#     # tf.initialize_all_variables() no long valid from
#     # 2017-03-02 if using tensorflow >= 0.12
#     if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
#         init = tf.initialize_all_variables()
#     else:
#         init = tf.global_variables_initializer()
#     sess.run(init)
#     step = 0
#     while step * batch_size < training_iters:
#         batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#         batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
#         _, acc, loss = sess.run([train_op, accuracy, cost], feed_dict={
#             x: batch_xs,
#             y: batch_ys,
#         })
#         if step % 20 == 0:
#             print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
#                   "{:.6f}".format(loss) + ", Training Accuracy= " + \
#                   "{:.5f}".format(acc))
#         step += 1



import tensorflow as tf
import numpy as np
import os

tf.set_random_seed(777)  # reproducibility



def MinMaxScaler(data):
    ''' Min Max Normalization
    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]
    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]
    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    print(data.shape)
    return numerator / (denominator + 1e-7)



# train Parameters
seq_length = 6
data_dim = 7
hidden_dim = 5
output_dim = 1
learning_rate = 0.01
iterations = 500

# Open, High, Low, Volume, Close
# high  diff_24h   diff_per_24h	 bid  ask	 low    volume   last
xy = np.loadtxt('D:\\data\\bitcoin_okcoin_usd222.csv', delimiter=',')

xy = xy[::-1]  # reverse order (chronically ordered)
xy = MinMaxScaler(xy)
x = xy
y = xy[:, [-1]]  # last as label



# build a dataset
dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]  # Next last price
    print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)

# train/test split
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])

# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

# build a LSTM network
cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

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
#
