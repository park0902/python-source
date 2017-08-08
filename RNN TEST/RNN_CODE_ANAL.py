import tensorflow as tf
import numpy as np
import pprint as pp
'''
one hot encoding
h = [1,0,0,0]
e = [0,1,0,0]
l = [0,0,1,0]
o = [0,0,0,1]

input = [[[1,0,0,0]]]
shape = (1,1,4) => input demension : 4

hidden_size = 2 이면

output = [[[x,x]]]
shape = (1,1,2) => output demension : 4
'''

hidden_size = 2
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)

x_data = np.array([[[1,0,0,0]]], dtype=np.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    pp.pprint(outputs.eval())
    # array([[[ 0.08988617, -0.12192202]]], dtype=float32)


'''
Unfolding to n sequence

hidden_size = 2
sequence_length = 5


output_shape = (1,5,2) : [[[x,x], [x,x], [x,x], [x,x], [x,x]]]
input_shape = (1,5,4) : [[[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,1,0], [0,0,0,1]]] 

'''
import tensorflow as tf
import numpy as np
import pprint as pp

# one hot encoding
h = [1,0,0,0]
e = [0,1,0,0]
l = [0,0,1,0]
o = [0,0,0,1]

# one cell RNN input_dim(4) -> output_dim(2).  sequence : 5

hidden_size = 2
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
x_data = np.array([[h,e,l,l,o]], dtype=np.float32)
#                sequence_size = 5

print(x_data.shape)
# (1, 5, 4)

pp.pprint(x_data)
# array([[[ 1.,  0.,  0.,  0.],
#         [ 0.,  1.,  0.,  0.],
#         [ 0.,  0.,  1.,  0.],
#         [ 0.,  0.,  1.,  0.],
#         [ 0.,  0.,  0.,  1.]]], dtype=float32)

outputs, states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    pp.pprint(outputs.eval())
    # array([[[ 0.01597848, -0.10382184],
#             [ 0.10696401, -0.13252695],
#             [-0.00212043, -0.17253044],
#             [-0.12693863, -0.20007074],
#             [-0.21488167, -0.12674101]]], dtype=float32)




'''
Batching input

hidden_size = 2
sequence_length = 5
batch_size = 3

output_shape = (3,5,2)
input_shape = (3,5,4)
'''
import tensorflow as tf
import numpy as np
import pprint as pp

# one hot encoding
h = [1,0,0,0]
e = [0,1,0,0]
l = [0,0,1,0]
o = [0,0,0,1]

x_data = np.array([[h,e,l,l,o],
                   [e,o,l,l,l],
                   [l,l,e,e,l]], dtype=np.float32)

pp.pprint(x_data)

cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=2, state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    pp.pprint(outputs.eval())



'''
Hi Hello Training
'''
import tensorflow as tf
import numpy as np

hidden_size = 5         # output from the LSTM
input_dim = 5           # one-hot size
batch_size = 1          # one sentence
sequence_length = 6     # |ihello| == 6



idx2char = ['h', 'i', 'e', 'l', 'o']
x_data = [[0,1,0,2,3,3]]    # hihell
x_one_hot = [[[1,0,0,0,0],
              [0,1,0,0,0],
              [1,0,0,0,0],
              [0,0,1,0,0],
              [0,0,0,1,0],
              [0,0,0,1,0]]]

y_data = [[1,0,2,3,3,4]]    # ihello

X = tf.placeholder(tf.float32, [None, sequence_length, input_dim])  # X data
Y = tf.placeholder(tf.float32, [None, sequence_length])  # Y label

cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y,
                                                 weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        l, _ = sess.run([loss, train],feed_dict={X: x_one_hot, Y:y_data})
        result = sess.run(prediction, feed_dict={X: x_one_hot})
        print(i, "loss : ", l, "preditcion : ", result, "True Y : ", y_data)

        # result_str = [idx2char[c] for c in np.squeeze(result)]
        # print("\tPrediction str : ", ''.join(result_str))






import os

file_list = os.listdir('data/')

print(file_list)




