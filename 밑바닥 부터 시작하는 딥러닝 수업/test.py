# # from tensorflow.python.training import moving_averages as ema
# # import tensorflow as tf
# #
# # def batch_norm(self, input, shape, training, convl=True, name='BN', decay=0.99):
# #     beta = tf.Variable(tf.constant(0.0, shape=[shape]), name='beta')
# #     scale = tf.Variable(tf.constant(1.0, shape=[shape]), name='scale')
# #     moving_collections = ['moving_variables', tf.GraphKeys.MOVING_AVERAGE_VARIABLES]
# #     moving_mean = tf.Variable(tf.zeros([shape]), trainable=False, collections=moving_collections, name='moving_mean')
# #     moving_var = tf.Variable(tf.ones([shape]), trainable=False, collections=moving_collections, name='moving_var')
# #
# #     if training is True:
# #         if convl:
# #             mean, var = tf.nn.moments(input, [0, 1, 2], name='moments')
# #         else:
# #             mean, var = tf.nn.moments(input, [0], name='moments')
# #
# #         update_moving_mean = ema.assign_moving_average(moving_mean, mean, decay)
# #         tf.add_to_collection('_update_ops_', update_moving_mean)
# #         update_moving_var = ema.assign_moving_average(moving_var, var, decay)
# #         tf.add_to_collection('_update_ops_', update_moving_var)
# #         tf.assign_sub()
# #     else:
# #         mean, var = moving_mean, moving_var
# #
# #     output = tf.nn.batch_normalization(input, mean, var, beta, scale, variance_epsilon=1e-3, name=name)
# #     output.set_shape(input.get_shape())
# #     return output
# #
# #
# #
# #
# #
# #
# #
# # from tensorflow.python.training import moving_averages as ema
# # import tensorflow as tf
# #
# # def batch_norm(self, input, shape, training, convl=True, name='BN', decay=0.99):
# #     beta = tf.Variable(tf.constant(0.0, shape=[shape]), name='beta')
# #     scale = tf.Variable(tf.constant(1.0, shape=[shape]), name='scale')
# #     moving_collections = ['moving_variables', tf.GraphKeys.MOVING_AVERAGE_VARIABLES]
# #     moving_mean = tf.Variable(tf.zeros([shape]), trainable=False, collections=moving_collections, name='moving_mean')
# #     moving_var = tf.Variable(tf.ones([shape]), trainable=False, collections=moving_collections, name='moving_var')
# #
# #     if training is True:
# #         if convl:
# #             mean, var = tf.nn.moments(input, [0, 1, 2], name='moments')
# #         else:
# #             mean, var = tf.nn.moments(input, [0], name='moments')
# #
# #         update_moving_mean = tf.assign_sub(moving_mean, mean)
# #         tf.add_to_collection('_update_ops_', update_moving_mean)
# #         update_moving_var = tf.assign_sub(moving_var, var)
# #         tf.add_to_collection('_update_ops_', update_moving_var)
# #         tf.assign_sub()
# #     else:
# #         mean, var = moving_mean, moving_var
# #
# #     output = tf.nn.batch_normalization(input, mean, var, beta, scale, variance_epsilon=1e-3, name=name)
# #     output.set_shape(input.get_shape())
# #     return output
#
#
#
# """
# Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
# BSD License
# """
# import numpy as np
#
# # data I/O
# data = open('D:\\data\\winter.txt', 'r').read()  # should be simple plain text file
# chars = list(set(data))
# data_size, vocab_size = len(data), len(chars)
# print('data has %d characters, %d unique.' % (data_size, vocab_size))
# char_to_ix = {ch: i for i, ch in enumerate(chars)}
# ix_to_char = {i: ch for i, ch in enumerate(chars)}
#
# # hyperparameters
# hidden_size = 100  # size of hidden layer of neurons
# seq_length = 25  # number of steps to unroll the RNN for
# learning_rate = 1e-1
#
# # model parameters
# Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  # input to hidden
# Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
# Why = np.random.randn(vocab_size, hidden_size) * 0.01  # hidden to output
# bh = np.zeros((hidden_size, 1))  # hidden bias
# by = np.zeros((vocab_size, 1))  # output bias
#
#
# def lossFun(inputs, targets, hprev):
#     """
#     inputs,targets are both list of integers.
#     hprev is Hx1 array of initial hidden state
#     returns the loss, gradients on model parameters, and last hidden state
#     """
#     xs, hs, ys, ps = {}, {}, {}, {}
#     hs[-1] = np.copy(hprev)
#     loss = 0
#     # forward pass
#     for t in range(len(inputs)):
#         xs[t] = np.zeros((vocab_size, 1))  # encode in 1-of-k representation
#         xs[t][inputs[t]] = 1
#         hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t - 1]) + bh)  # hidden state
#         ys[t] = np.dot(Why, hs[t]) + by  # unnormalized log probabilities for next chars
#         ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # probabilities for next chars
#         loss += -np.log(ps[t][targets[t], 0])  # softmax (cross-entropy loss)
#     # backward pass: compute gradients going backwards
#     dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
#     dbh, dby = np.zeros_like(bh), np.zeros_like(by)
#     dhnext = np.zeros_like(hs[0])
#     for t in reversed(range(len(inputs))):
#         dy = np.copy(ps[t])
#         dy[targets[
#             t]] -= 1  # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
#         dWhy += np.dot(dy, hs[t].T)
#         dby += dy
#         dh = np.dot(Why.T, dy) + dhnext  # backprop into h
#         dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh nonlinearity
#         dbh += dhraw
#         dWxh += np.dot(dhraw, xs[t].T)
#         dWhh += np.dot(dhraw, hs[t - 1].T)
#         dhnext = np.dot(Whh.T, dhraw)
#     for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
#         np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients
#     return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]
#
#
# def sample(h, seed_ix, n):
#     """
#     sample a sequence of integers from the model
#     h is memory state, seed_ix is seed letter for first time step
#     """
#     x = np.zeros((vocab_size, 1))
#     x[seed_ix] = 1
#     ixes = []
#     for t in range(n):
#         h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
#         y = np.dot(Why, h) + by
#         p = np.exp(y) / np.sum(np.exp(y))
#         ix = np.random.choice(range(vocab_size), p=p.ravel())
#         x = np.zeros((vocab_size, 1))
#         x[ix] = 1
#         ixes.append(ix)
#     return ixes
#
#
# n, p = 0, 0
# mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
# mbh, mby = np.zeros_like(bh), np.zeros_like(by)  # memory variables for Adagrad
# smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # loss at iteration 0
# while True:
#     # prepare inputs (we're sweeping from left to right in steps seq_length long)
#     if p + seq_length + 1 >= len(data) or n == 0:
#         hprev = np.zeros((hidden_size, 1))  # reset RNN memory
#         p = 0  # go from start of data
#     inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]
#     targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]
#
#     # sample from the model now and then
#     if n % 100 == 0:
#         sample_ix = sample(hprev, inputs[0], 200)
#         txt = ''.join(ix_to_char[ix] for ix in sample_ix)
#         print('----\n %s \n----' % (txt,))
#
#     # forward seq_length characters through the net and fetch gradient
#     loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
#     smooth_loss = smooth_loss * 0.999 + loss * 0.001
#     if n % 100 == 0:
#         print('iter %d, loss: %f' % (n, smooth_loss))  # print progress
#
#     # perform parameter update with Adagrad
#     for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
#                                   [dWxh, dWhh, dWhy, dbh, dby],
#                                   [mWxh, mWhh, mWhy, mbh, mby]):
#         mem += dparam * dparam
#         param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update
#
#     p += seq_length  # move data pointer
#     n += 1  # iteration counter



########################################################################################################################
# #implemented as I read Andrej Karpathy's post on RNNs.
# import numpy as np
# import matplotlib.pyplot as plt
#
# class RNN(object):
#
#     def __init__(self, insize, outsize, hidsize, learning_rate):
#         self.insize = insize
#
#         self.h = np.zeros((hidsize , 1))#a [h x 1] hidden state stored from last batch of inputs
#
#         #parameters
#         self.W_hh = np.random.randn(hidsize, hidsize)*0.01#[h x h]
#         self.W_xh = np.random.randn(hidsize, insize)*0.01#[h x x]
#         self.W_hy = np.random.randn(outsize, hidsize)*0.01#[y x h]
#         self.b_h = np.zeros((hidsize, 1))#biases
#         self.b_y = np.zeros((outsize, 1))
#
#         #the Adagrad gradient update relies upon having a memory of the sum of squares of dparams
#         self.adaW_hh = np.zeros((hidsize, hidsize))
#         self.adaW_xh = np.zeros((hidsize, insize))
#         self.adaW_hy = np.zeros((outsize, hidsize))
#         self.adab_h = np.zeros((hidsize, 1))
#         self.adab_y = np.zeros((outsize, 1))
#
#         self.learning_rate = learning_rate
#
#     #give the RNN a sequence of inputs and outputs (seq_length long), and use
#     #them to adjust the internal state
#     def train(self, x, y):
#         #=====initialize=====
#         xhat = {}#holds 1-of-k representations of x
#         yhat = {}#holds 1-of-k representations of predicted y (unnormalized log probs)
#         p = {}#the normalized probabilities of each output through time
#         h = {}#holds state vectors through time
#         h[-1] = np.copy(self.h)#we will need to access the previous state to calculate the current state
#
#         dW_xh = np.zeros_like(self.W_xh)
#         dW_hh = np.zeros_like(self.W_hh)
#         dW_hy = np.zeros_like(self.W_hy)
#         db_h = np.zeros_like(self.b_h)
#         db_y = np.zeros_like(self.b_y)
#         dh_next = np.zeros_like(self.h)
#
#         #=====forward pass=====
#         loss = 0
#         for t in range(len(x)):
#             xhat[t] = np.zeros((self.insize, 1))
#             xhat[t][x[t]] = 1#xhat[t] = 1-of-k representation of x[t]
#
#             h[t] = np.tanh(np.dot(self.W_xh, xhat[t]) + np.dot(self.W_hh, h[t-1]) + self.b_h)#find new hidden state
#             yhat[t] = np.dot(self.W_hy, h[t]) + self.b_y#find unnormalized log probabilities for next chars
#
#             p[t] = np.exp(yhat[t]) / np.sum(np.exp(yhat[t]))#find probabilities for next chars
#
#             loss += -np.log(p[t][y[t],0])#softmax (cross-entropy loss)
#
#         #=====backward pass: compute gradients going backwards=====
#         for t in reversed(range(len(x))):
#             #backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
#             dy = np.copy(p[t])
#             dy[y[t]] -= 1
#
#             #find updates for y
#             dW_hy += np.dot(dy, h[t].T)
#             db_y += dy
#
#             #backprop into h and through tanh nonlinearity
#             dh = np.dot(self.W_hy.T, dy) + dh_next
#             dh_raw = (1 - h[t]**2) * dh
#
#             #find updates for h
#             dW_xh += np.dot(dh_raw, xhat[t].T)
#             dW_hh += np.dot(dh_raw, h[t-1].T)
#             db_h += dh_raw
#
#             #save dh_next for subsequent iteration
#             dh_next = np.dot(self.W_hh.T, dh_raw)
#
#         for dparam in [dW_xh, dW_hh, dW_hy, db_h, db_y]:
#             np.clip(dparam, -5, 5, out=dparam)#clip to mitigate exploding gradients
#
#         #update RNN parameters according to Adagrad
#         for param, dparam, adaparam in zip([self.W_hh, self.W_xh, self.W_hy, self.b_h, self.b_y], \
#                                 [dW_hh, dW_xh, dW_hy, db_h, db_y], \
#                                 [self.adaW_hh, self.adaW_xh, self.adaW_hy, self.adab_h, self.adab_y]):
#             adaparam += dparam*dparam
#             param += -self.learning_rate*dparam/np.sqrt(adaparam+1e-8)
#
#         self.h = h[len(x)-1]
#
#         return loss
#
#     #let the RNN generate text
#     def sample(self, seed, n):
#         ndxs = []
#         h = self.h
#
#         xhat = np.zeros((self.insize, 1))
#         xhat[seed] = 1#transform to 1-of-k
#
#         for t in range(n):
#             h = np.tanh(np.dot(self.W_xh, xhat) + np.dot(self.W_hh, h) + self.b_h)#update the state
#             y = np.dot(self.W_hy, h) + self.b_y
#             p = np.exp(y) / np.sum(np.exp(y))
#             ndx = np.random.choice(range(self.insize), p=p.ravel())
#
#             xhat = np.zeros((self.insize, 1))
#             xhat[ndx] = 1
#
#             ndxs.append(ndx)
#
#         return ndxs
#
#
# def test():
#     #open a text file
#     data = open('D:\\data\\winter1.txt', 'r').read() # should be simple plain text file
#     print(len(data))
#     chars = list(set(data))
#     data_size, vocab_size = len(data), len(chars)
#     print('data has %d characters, %d unique.' % (data_size, vocab_size))
#
#     #make some dictionaries for encoding and decoding from 1-of-k
#     char_to_ix = { ch:i for i,ch in enumerate(chars) }
#     ix_to_char = { i:ch for i,ch in enumerate(chars) }
#
#     #insize and outsize are len(chars). hidsize is 100. seq_length is 25. learning_rate is 0.1.
#     rnn = RNN(len(chars), len(chars), 100, 0.1)
#
#     #iterate over batches of input and target output
#     seq_length = 25
#     losses = []
#     smooth_loss = -np.log(1.0/len(chars))*seq_length#loss at iteration 0
#     losses.append(smooth_loss)
#
#     for i in range(int(len(data)/seq_length)):
#         x = [char_to_ix[c] for c in data[i*seq_length:(i+1)*seq_length]]#inputs to the RNN
#         y = [char_to_ix[c] for c in data[i*seq_length+1:(i+1)*seq_length+1]]#the targets it should be outputting
#
#         if i%1000==0:
#             sample_ix = rnn.sample(x[0], 200)
#             txt = ''.join([ix_to_char[n] for n in sample_ix])
#             print(txt)
#
#         loss = rnn.train(x, y)
#         smooth_loss = smooth_loss*0.999 + loss*0.001
#
#         if i%1000==0:
#             print('iteration %d, smooth_loss = %f' % (i, smooth_loss))
#             losses.append(smooth_loss)
#
#     plt.plot(range(len(losses)), losses, 'b', label='smooth loss')
#     plt.xlabel('time in thousands of iterations')
#     plt.ylabel('loss')
#     plt.legend()
#     plt.show()
#
# if __name__ == "__main__":
#     test()



########################################################################################################################
#An attempt at a batched RNNs
#
#I don't think this is an LSTM. What is the difference, exactly? I want to
#know the more complicated functional forms, how to backprop them, and what
#the advantage is.
import numpy as np
import matplotlib.pyplot as plt

class RNNlayer(object):

    def __init__(self, x_size, h_size, y_size, learning_rate):
        self.h_size = h_size
        self.learning_rate = learning_rate#ugh, nightmares

        #inputs and internal states for each layer, used during backpropagation
        self.x = {}
        self.h = {}
        self.h_last = np.zeros((h_size, 1))

        #x is the input. h is the internal hidden stuff. y is the output.
        self.W_xh = np.random.randn(h_size, x_size)*0.01#x -> h
        self.W_hh = np.random.randn(h_size, h_size)*0.01#h -> h
        self.W_hy = np.random.randn(y_size, h_size)*0.01#h -> y
        self.b_h = np.zeros((h_size, 1))#biases
        self.b_y = np.zeros((y_size, 1))

        #the Adagrad gradient update relies upon having a memory of the sum of squares of dparams
        self.adaW_xh = np.zeros((h_size, x_size))#start sums at 0
        self.adaW_hh = np.zeros((h_size, h_size))
        self.adaW_hy = np.zeros((y_size, h_size))
        self.adab_h = np.zeros((h_size, 1))
        self.adab_y = np.zeros((y_size, 1))

    #given an input, step the internal state and return the output of the network
    #Because the whole network is together in one object, I can make it easy and just
    #take a list of input ints, transform them to 1-of-k once, and prop everywhere.
    #
    #   Here is a diagram of what's happening. Useful to understand backprop too.
    #
    #                  [b_h]                                              [b_y]
    #                    v                                                  v
    #   x -> [W_xh] -> [sum] -> h_raw -> [nonlinearity] -> h -> [W_hy] -> [sum] -> y ... -> [e] -> p
    #                    ^                                 |
    #                    '----h_next------[W_hh]-----------'
    #
    def step(self, x):
        #load the last state from the last batch in to the beginning of h
        #it is necessary to save it outside of h because h is used in backprop
        self.h[-1] = self.h_last
        self.x = x

        y = {}
        p = {}#p[t] = the probabilities of next chars given chars passed in at times <=t
        for t in range(len(self.x)):#for each moment in time

            #self.h[t] = np.maximum(0, np.dot(self.W_xh, self.xhat[t]) + \
            #   np.dot(self.W_hh, self.h[t-1]) + self.b_h)#ReLU

            #find new hidden state in this layer at this time
            self.h[t] = np.tanh(np.dot(self.W_xh, self.x[t]) + \
                np.dot(self.W_hh, self.h[t-1]) + self.b_h)#tanh

            #find unnormalized log probabilities for next chars
            y[t] = np.dot(self.W_hy, self.h[t]) + self.b_y#output from this layer is input to the next
            p[t] = np.exp(y[t]) / np.sum(np.exp(y[t]))#find probabilities for next chars

        #save the last state from this batch for next batch
        self.h_last = self.h[len(x)-1]

        return y, p

    #given the RNN a sequence of correct outputs (seq_length long), use
    #them and the internal state to adjust weights
    def backprop(self, dy):

        #we will need some place to store gradients
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)

        dh_next = np.zeros((self.h_size, 1))#I think this is the right dimension
        dx = {}

        for t in reversed(range(len(dy))):
            #find updates for y stuff
            dW_hy += np.dot(dy[t], self.h[t].T)
            db_y += dy[t]

            #backprop into h and through nonlinearity
            dh = np.dot(self.W_hy.T, dy[t]) + dh_next
            dh_raw = (1 - self.h[t]**2)*dh#tanh
            #dh_raw = self.h[t][self.h[t] <= 0] = 0#ReLU

            #find updates for h stuff
            dW_xh += np.dot(dh_raw, self.x[t].T)
            dW_hh += np.dot(dh_raw, self.h[t-1].T)
            db_h += dh_raw

            #save dh_next for subsequent iteration
            dh_next = np.dot(self.W_hh.T, dh_raw)

            #save the error to propagate to the next layer. Am I doing this correctly?
            dx[t] = np.dot(self.W_xh.T, dh_raw)

        #clip to mitigate exploding gradients
        for dparam in [dW_xh, dW_hh, dW_hy, db_h, db_y]:
            dparam = np.clip(dparam, -5, 5)
        for t in range(len(dx)):
            dx[t] = np.clip(dx[t], -5, 5)

        #update RNN parameters according to Adagrad
        #yes, it calls by reference, so the actual things do get updated
        for param, dparam, adaparam in zip([self.W_hh, self.W_xh, self.W_hy, self.b_h, self.b_y], \
                    [dW_hh, dW_xh, dW_hy, db_h, db_y], \
                    [self.adaW_hh, self.adaW_xh, self.adaW_hy, self.adab_h, self.adab_y]):
            adaparam += dparam*dparam
            param += -self.learning_rate*dparam/np.sqrt(adaparam+1e-8)

        return dx

def test():
    #open a text file
    data = open('D:\\data\\해리포터.txt', 'r').read() # should be simple plain text file
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    print('data has %d characters, %d unique.' % (data_size, vocab_size))

    #make some dictionaries for encoding and decoding from 1-of-k
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }

    #num_hid_layers = 3, insize and outsize are len(chars). hidsize is 512 for all layers. learning_rate is 0.1.
    rnn1 = RNNlayer(len(chars), 50, 50, 0.001)
    rnn2 = RNNlayer(50, 50, 50, 0.001)
    rnn3 = RNNlayer(50, 50, len(chars), 0.001)

    #iterate over batches of input and target output
    seq_length = 25
    losses = []
    smooth_loss = -np.log(1.0/len(chars))*seq_length#loss at iteration 0
    losses.append(smooth_loss)
    smooth_error = seq_length

    for j in range(50):
        print("============== j = ",j," ==================")
        for i in range(int(len(data)/(seq_length))):
            inputs = [char_to_ix[c] for c in data[i*seq_length:(i+1)*seq_length]]#inputs to the RNN
            targets = [char_to_ix[c] for c in data[i*seq_length+1:(i+1)*seq_length+1]]#the targets it should be outputting

            if i%1000==0:
                sample_ix = sample([rnn1, rnn2, rnn3], inputs[0], 200, len(chars))
                txt = ''.join([ix_to_char[n] for n in sample_ix])
                print(txt)
                losses.append(smooth_loss)

            #forward pass
            x = oneofk(inputs, len(chars))
            y1, p1 = rnn1.step(x)
            y2, p2 = rnn2.step(y1)
            y3, p3 = rnn3.step(y2)

            #calculate loss and error rate
            loss = 0
            error = 0
            for t in range(len(targets)):
                loss += -np.log(p3[t][targets[t],0])
                if np.argmax(p3[t]) != targets[t]:
                    error += 1
            smooth_loss = smooth_loss*0.999 + loss*0.001
            smooth_error = smooth_error*0.999 + error*0.001

            if i%10==0:
                print(i,"\tsmooth loss =",smooth_loss,"\tsmooth error rate =",float(smooth_error)/len(targets))

            #backward pass
            dy = logprobs(p3, targets)
            dx3 = rnn3.backprop(dy)
            dx2 = rnn2.backprop(dx3)
            dx1 = rnn1.backprop(dx2)

    plt.plot(range(len(losses)), losses, 'b', label='smooth loss')
    plt.xlabel('time in thousands of iterations')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


#let the RNN generate text
def sample(rnns, seed, n, k):

    ndxs = []
    ndx = seed

    for t in range(n):
        x = oneofk([ndx], k)
        for i in range(len(rnns)):
            x, p = rnns[i].step(x)

        ndx = np.random.choice(range(len(p[0])), p=p[0].ravel())
        ndxs.append(ndx)

    return ndxs

#I have these out here because it's not really the RNN's concern how you transform
#things to a form it can understand

#get the initial dy to pass back through the first layer
def logprobs(p, targets):
    dy = {}
    for t in range(len(targets)):
        #see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
        dy[t] = np.copy(p[t])
        dy[t][targets[t]] -= 1
    return dy

#encode inputs in 1-of-k so they match inputs between layers
def oneofk(inputs, k):
    x = {}
    for t in range(len(inputs)):
        x[t] = np.zeros((k, 1))#initialize x input to 1st hidden layer
        x[t][inputs[t]] = 1#it's encoded in 1-of-k representation
    return x

if __name__ == "__main__":
    test()