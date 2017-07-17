# import numpy as np, tensorflow as tf, tqdm
# from tensorflow.examples.tutorials.mnist import input_data
# import matplotlib.pyplot as plt
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#
#
# # Generate predetermined random weights so the networks are similarly initialized
# w1_initial = np.random.normal(size=(784,100)).astype(np.float32)
# w2_initial = np.random.normal(size=(100,100)).astype(np.float32)
# w3_initial = np.random.normal(size=(100,10)).astype(np.float32)
#
# # Small epsilon value for the BN transform
# epsilon = 1e-3
#
#
# # Placeholders
# x = tf.placeholder(tf.float32, shape=[None, 784])
# y_ = tf.placeholder(tf.float32, shape=[None, 10])
# # Layer 1 without BN
# w1 = tf.Variable(w1_initial)
# b1 = tf.Variable(tf.zeros([100]))
# z1 = tf.matmul(x,w1)+b1
# l1 = tf.nn.sigmoid(z1)
#
#
# # Layer 1 with BN
# w1_BN = tf.Variable(w1_initial)
#
# # Note that pre-batch normalization bias is ommitted. The effect of this bias would be
# # eliminated when subtracting the batch mean. Instead, the role of the bias is performed
# # by the new beta variable. See Section 3.2 of the BN2015 paper.
# z1_BN = tf.matmul(x,w1_BN)
#
# # Calculate batch mean and variance
# batch_mean1, batch_var1 = tf.nn.moments(z1_BN,[0])
#
# # Apply the initial batch normalizing transform
# z1_hat = (z1_BN - batch_mean1) / tf.sqrt(batch_var1 + epsilon)
#
# # Create two new parameters, scale and beta (shift)
# scale1 = tf.Variable(tf.ones([100]))
# beta1 = tf.Variable(tf.zeros([100]))
#
# # Scale and shift to obtain the final output of the batch normalization
# # this value is fed into the activation function (here a sigmoid)
# BN1 = scale1 * z1_hat + beta1
# l1_BN = tf.nn.sigmoid(BN1)
# # Layer 2 without BN
# w2 = tf.Variable(w2_initial)
# b2 = tf.Variable(tf.zeros([100]))
# z2 = tf.matmul(l1,w2)+b2
# l2 = tf.nn.sigmoid(z2)
#
#
#
# # Layer 2 with BN, using Tensorflows built-in BN function
# w2_BN = tf.Variable(w2_initial)
# z2_BN = tf.matmul(l1_BN,w2_BN)
# batch_mean2, batch_var2 = tf.nn.moments(z2_BN,[0])
# scale2 = tf.Variable(tf.ones([100]))
# beta2 = tf.Variable(tf.zeros([100]))
# BN2 = tf.nn.batch_normalization(z2_BN,batch_mean2,batch_var2,beta2,scale2,epsilon)
# l2_BN = tf.nn.sigmoid(BN2)
# # Softmax
# w3 = tf.Variable(w3_initial)
# b3 = tf.Variable(tf.zeros([10]))
# y  = tf.nn.softmax(tf.matmul(l2,w3)+b3)
#
# w3_BN = tf.Variable(w3_initial)
# b3_BN = tf.Variable(tf.zeros([10]))
# y_BN  = tf.nn.softmax(tf.matmul(l2_BN,w3_BN)+b3_BN)
# # Loss, optimizer and predictions
# cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# cross_entropy_BN = -tf.reduce_sum(y_*tf.log(y_BN))
#
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# train_step_BN = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy_BN)
#
# correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
# correct_prediction_BN = tf.equal(tf.arg_max(y_BN,1),tf.arg_max(y_,1))
# accuracy_BN = tf.reduce_mean(tf.cast(correct_prediction_BN,tf.float32))
#
# zs, BNs, acc, acc_BN = [], [], [], []
#
# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
#
# for i in tqdm.tqdm(range(40000)):
#     batch = mnist.train.next_batch(60)
#     train_step.run(feed_dict={x: batch[0], y_: batch[1]})
#     train_step_BN.run(feed_dict={x: batch[0], y_: batch[1]})
#     if i % 50 is 0:
#         res = sess.run([accuracy,accuracy_BN,z2,BN2],feed_dict={x: mnist.test.images, y_: mnist.test.labels})
#         acc.append(res[0])
#         acc_BN.append(res[1])
#         zs.append(np.mean(res[2],axis=0)) # record the mean value of z2 over the entire test set
#         BNs.append(np.mean(res[3],axis=0)) # record the mean value of BN2 over the entire test set
#
# zs, BNs, acc, acc_BN = np.array(zs), np.array(BNs), np.array(acc), np.array(acc_BN)
#
#
#
# fig, ax = plt.subplots()
#
# ax.plot(range(0,len(acc)*50,50),acc, label='Without BN')
# ax.plot(range(0,len(acc)*50,50),acc_BN, label='With BN')
# ax.set_xlabel('Training steps')
# ax.set_ylabel('Accuracy')
# ax.set_ylim([0.8,1])
# ax.set_title('Batch Normalization Accuracy')
# ax.legend(loc=4)
# plt.show()


#######################################################################################
#
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# import os
# import numpy as np
# import tensorflow as tf
#
# from tensorflow.examples.tutorials.mnist import input_data
# from my_nn_lib import Convolution2D, MaxPooling2D
# from my_nn_lib import FullConnected, ReadOutLayer
#
# mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
# chkpt_file = '../MNIST_data/mnist_cnn.ckpt'
#
#
# def batch_norm(x, n_out, phase_train):
#     """
#     Batch normalization on convolutional maps.
#     Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
#     Args:
#         x:           Tensor, 4D BHWD input maps
#         n_out:       integer, depth of input maps
#         phase_train: boolean tf.Varialbe, true indicates training phase
#         scope:       string, variable scope
#     Return:
#         normed:      batch-normalized maps
#     """
#     with tf.variable_scope('bn'):
#         beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
#                            name='beta', trainable=True)
#         gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
#                             name='gamma', trainable=True)
#         batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
#         ema = tf.train.ExponentialMovingAverage(decay=0.5)
#
#         def mean_var_with_update():
#             ema_apply_op = ema.apply([batch_mean, batch_var])
#             with tf.control_dependencies([ema_apply_op]):
#                 return tf.identity(batch_mean), tf.identity(batch_var)
#
#         mean, var = tf.cond(phase_train,
#                             mean_var_with_update,
#                             lambda: (ema.average(batch_mean), ema.average(batch_var)))
#         normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
#     return normed
#
#
# #
#
# def training(loss, learning_rate):
#     optimizer = tf.train.AdamOptimizer(learning_rate)
#     # Create a variable to track the global step.
#     global_step = tf.Variable(0, name='global_step', trainable=False)
#     train_op = optimizer.minimize(loss, global_step=global_step)
#
#     return train_op
#
#
# def evaluation(y_pred, y):
#     correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
#
#     return accuracy
#
#
# def mlogloss(predicted, actual):
#     '''
#       args.
#          predicted : predicted probability
#                     (sum of predicted proba should be 1.0)
#          actual    : actual value, label
#     '''
#
#     def inner_fn(item):
#         eps = 1.e-15
#         item1 = min(item, (1 - eps))
#         item1 = max(item, eps)
#         res = np.log(item1)
#
#         return res
#
#     nrow = actual.shape[0]
#     ncol = actual.shape[1]
#
#     mysum = sum([actual[i, j] * inner_fn(predicted[i, j])
#                  for i in range(nrow) for j in range(ncol)])
#
#     ans = -1 * mysum / nrow
#
#     return ans
#
#
# #
#
# # Create the model
# def inference(x, y_, keep_prob, phase_train):
#     x_image = tf.reshape(x, [-1, 28, 28, 1])
#
#     with tf.variable_scope('conv_1'):
#         conv1 = Convolution2D(x, (28, 28), 1, 32, (5, 5), activation='none')
#         conv1_bn = batch_norm(conv1.output(), 32, phase_train)
#         conv1_out = tf.nn.relu(conv1_bn)
#
#         pool1 = MaxPooling2D(conv1_out)
#         pool1_out = pool1.output()
#
#     with tf.variable_scope('conv_2'):
#         conv2 = Convolution2D(pool1_out, (28, 28), 32, 64, (5, 5),
#                               activation='none')
#         conv2_bn = batch_norm(conv2.output(), 64, phase_train)
#         conv2_out = tf.nn.relu(conv2_bn)
#
#         pool2 = MaxPooling2D(conv2_out)
#         pool2_out = pool2.output()
#         pool2_flat = tf.reshape(pool2_out, [-1, 7 * 7 * 64])
#
#     with tf.variable_scope('fc1'):
#         fc1 = FullConnected(pool2_flat, 7 * 7 * 64, 1024)
#         fc1_out = fc1.output()
#         fc1_dropped = tf.nn.dropout(fc1_out, keep_prob)
#
#     y_pred = ReadOutLayer(fc1_dropped, 1024, 10).output()
#
#     cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_pred),
#                                                   reduction_indices=[1]))
#     loss = cross_entropy
#     train_step = training(loss, 1.e-4)
#     accuracy = evaluation(y_pred, y_)
#
#     return loss, accuracy, y_pred
#
#
# #
# if __name__ == '__main__':
#     TASK = 'train'  # 'train' or 'test'
#
#     # Variables
#     x = tf.placeholder(tf.float32, [None, 784])
#     y_ = tf.placeholder(tf.float32, [None, 10])
#     keep_prob = tf.placeholder(tf.float32)
#     phase_train = tf.placeholder(tf.bool, name='phase_train')
#
#     loss, accuracy, y_pred = inference(x, y_,
#                                        keep_prob, phase_train)
#
#     # Train
#     lr = 0.01
#     train_step = tf.train.AdagradOptimizer(lr).minimize(loss)
#     vars_to_train = tf.trainable_variables()  # option-1
#     vars_for_bn1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,  # TF >1.0
#                                      scope='conv_1/bn')
#     vars_for_bn2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,  # TF >1.0
#                                      scope='conv_2/bn')
#     vars_to_train = list(set(vars_to_train).union(set(vars_for_bn1)))
#     vars_to_train = list(set(vars_to_train).union(set(vars_for_bn2)))
#
#     if TASK == 'test' or os.path.exists(chkpt_file):
#         restore_call = True
#         vars_all = tf.all_variables()
#         vars_to_init = list(set(vars_all) - set(vars_to_train))
#         init = tf.variables_initializer(vars_to_init)  # TF >1.0
#     elif TASK == 'train':
#         restore_call = False
#         init = tf.global_variables_initializer()  # TF >1.0
#     else:
#         print('Check task switch.')
#
#     saver = tf.train.Saver(vars_to_train)  # option-1
#     # saver = tf.train.Saver()                   # option-2
#
#
#     with tf.Session() as sess:
#         # if TASK == 'train':              # add in option-2 case
#         sess.run(init)  # option-1
#
#         if restore_call:
#             # Restore variables from disk.
#             saver.restore(sess, chkpt_file)
#
#         if TASK == 'train':
#             print('\n Training...')
#             for i in range(5001):
#                 batch_xs, batch_ys = mnist.train.next_batch(100)
#                 train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.5,
#                                 phase_train: True})
#                 if i % 1000 == 0:
#                     cv_fd = {x: batch_xs, y_: batch_ys, keep_prob: 1.0,
#                              phase_train: False}
#                     train_loss = loss.eval(cv_fd)
#                     train_accuracy = accuracy.eval(cv_fd)
#
#                     print('  step, loss, accurary = %6d: %8.4f, %8.4f' % (i,
#                                                                           train_loss, train_accuracy))
#
#         # Test trained model
#         test_fd = {x: mnist.test.images, y_: mnist.test.labels,
#                    keep_prob: 1.0, phase_train: False}
#         print(' accuracy = %8.4f' % accuracy.eval(test_fd))
#         # Multiclass Log Loss
#         pred = y_pred.eval(test_fd)
#         act = mnist.test.labels
#         print(' multiclass logloss = %8.4f' % mlogloss(pred, act))
#
#         # Save the variables to disk.
#         if TASK == 'train':
#             save_path = saver.save(sess, chkpt_file)
#             print("Model saved in file: %s" % save_path)



########################################################################################################################


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import urllib

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
DATA_DIRECTORY = "data"
LOGS_DIRECTORY = "logs/train"

# train params
training_epochs = 15
batch_size = 100
display_step = 50

# network params
n_input = 784
n_hidden_1 = 256
n_hidden_2 = 256
n_classes = 10

# Store layers weight & bias

with tf.name_scope('weight'):
    normal_weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]),name='w1_normal'),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]),name='w2_normal'),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]),name='wout_normal')
    }
    truncated_normal_weights  = {
        'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],stddev=0.1),name='w1_truncated_normal'),
        'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],stddev=0.1),name='w2_truncated_normal'),
        'out': tf.Variable(tf.truncated_normal([n_hidden_2, n_classes],stddev=0.1),name='wout_truncated_normal')
    }
    xavier_weights  = {
        'h1': tf.get_variable('w1_xaiver', [n_input, n_hidden_1],initializer=tf.contrib.layers.xavier_initializer()),
        'h2': tf.get_variable('w2_xaiver', [n_hidden_1, n_hidden_2],initializer=tf.contrib.layers.xavier_initializer()),
        'out': tf.get_variable('wout_xaiver',[n_hidden_2, n_classes],initializer=tf.contrib.layers.xavier_initializer())
    }
    he_weights = {
        'h1': tf.get_variable('w1_he', [n_input, n_hidden_1],
                              initializer=tf.contrib.layers.variance_scaling_initializer()),
        'h2': tf.get_variable('w2_he', [n_hidden_1, n_hidden_2],
                              initializer=tf.contrib.layers.variance_scaling_initializer()),
        'out': tf.get_variable('wout_he', [n_hidden_2, n_classes],
                               initializer=tf.contrib.layers.variance_scaling_initializer())
    }
with tf.name_scope('bias'):
    normal_biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1]),name='b1_normal'),
        'b2': tf.Variable(tf.random_normal([n_hidden_2]),name='b2_normal'),
        'out': tf.Variable(tf.random_normal([n_classes]),name='bout_normal')
    }
    zero_biases = {
        'b1': tf.Variable(tf.zeros([n_hidden_1]),name='b1_zero'),
        'b2': tf.Variable(tf.zeros([n_hidden_2]),name='b2_zero'),
        'out': tf.Variable(tf.zeros([n_classes]),name='bout_normal')
    }
weight_initializer = {'normal':normal_weights, 'truncated_normal':truncated_normal_weights, 'xavier':xavier_weights, 'he':he_weights}
bias_initializer = {'normal':normal_biases, 'zero':zero_biases}

# user input
from argparse import ArgumentParser

WEIGHT_INIT = 'xavier'
BIAS_INIT = 'zero'
BACH_NORM = True

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--weight-init',
                        dest='weight_initializer', help='weight initializer',
                        metavar='WEIGHT_INIT', required=True)
    parser.add_argument('--bias-init',
                        dest='bias_initializer', help='bias initializer',
                        metavar='BIAS_INIT', required=True)
    parser.add_argument('--batch-norm',
                        dest='batch_normalization', help='boolean for activation of batch normalization',
                        metavar='BACH_NORM', required=True)
    return parser

# Download the data from Yann's website, unless it's already here.
def maybe_download(filename):
    if not tf.gfile.Exists(DATA_DIRECTORY):
        tf.gfile.MakeDirs(DATA_DIRECTORY)
    filepath = os.path.join(DATA_DIRECTORY, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath

# Batch normalization implementation
# from https://github.com/tensorflow/tensorflow/issues/1122
def batch_norm_layer(inputT, is_training=True, scope=None):
    # Note: is_training is tf.placeholder(tf.bool) type
    return tf.cond(is_training,
                    lambda: batch_norm(inputT, is_training=True,
                    center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9, scope=scope),
                    lambda: batch_norm(inputT, is_training=False,
                    center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9,
                    scope=scope, reuse = True))

# Create model of MLP with batch-normalization layer
def MLPwithBN(x, weights, biases, is_training=True):
    with tf.name_scope('MLPwithBN'):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = batch_norm_layer(layer_1,is_training=is_training, scope='layer_1_bn')
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = batch_norm_layer(layer_2, is_training=is_training, scope='layer_2_bn')
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Create model of MLP without batch-normalization layer
def MLPwoBN(x, weights, biases):
    with tf.name_scope('MLPwoBN'):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# main function
def main():
    # Parse argument
    parser = build_parser()
    options = parser.parse_args()
    weights = weight_initializer[options.weight_initializer]
    biases = bias_initializer[options.bias_initializer]
    batch_normalization = options.batch_normalization

    # Import data
    mnist = input_data.read_data_sets('data/', one_hot=True)

    # Boolean for MODE of train or test
    is_training = tf.placeholder(tf.bool, name='MODE')

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10]) #answer

    # Predict
    if batch_normalization=='True':
        y = MLPwithBN(x,weights,biases,is_training)
    else:
        y = MLPwoBN(x, weights, biases)

    # Get loss of model
    with tf.name_scope("LOSS"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y,y_))

    # Define optimizer
    with tf.name_scope("ADAM"):
        train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    # moving_mean and moving_variance need to be updated
    if batch_normalization == "True":
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            train_ops = [train_step] + update_ops
            train_op_final = tf.group(*train_ops)
        else:
            train_op_final = train_step

    # Get accuracy of model
    with tf.name_scope("ACC"):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Create a summary to monitor loss tensor
    tf.scalar_summary('loss', loss)

    # Create a summary to monitor accuracy tensor
    tf.scalar_summary('acc', accuracy)

    # Merge all summaries into a single op
    merged_summary_op = tf.merge_all_summaries()

    # Add ops to save and restore all the variables
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})

    # Training cycle
    total_batch = int(mnist.train.num_examples / batch_size)

    # op to write logs to Tensorboard
    summary_writer = tf.train.SummaryWriter(LOGS_DIRECTORY, graph=tf.get_default_graph())

    # Loop for epoch
    for epoch in range(training_epochs):

        # Loop over all batches
        for i in range(total_batch):

            batch = mnist.train.next_batch(batch_size)

            # Run optimization op (backprop), loss op (to get loss value)
            # and summary nodes
            _, train_accuracy, summary = sess.run([train_op_final, accuracy, merged_summary_op] , feed_dict={x: batch[0], y_: batch[1], is_training: True})

            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch * total_batch + i)

            # Display logs
            if i % display_step == 0:
                print("Epoch:", '%04d,' % (epoch + 1),
                "batch_index %4d/%4d, training accuracy %.5f" % (i, total_batch, train_accuracy))

    # Calculate accuracy for all mnist test images
    print("test accuracy for the latest result: %g" % accuracy.eval(
    feed_dict={x: mnist.test.images, y_: mnist.test.labels, is_training: False}))

if __name__ == '__main__':
    main()