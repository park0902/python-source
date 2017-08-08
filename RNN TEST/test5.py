import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def min_max_scaler(data):
    return (data - np.min(data, axis=0))/(np.max(data, axis=0) - np.min(data, axis=0) + 1e-5)

# DeepLearningProject/stock_prediction/data/
def read_data(file_name):
    data = np.loadtxt('d:\\data\\'+file_name, delimiter=',', skiprows=1)
    data = data[:, 1:]
    data = data[np.sum(np.isnan(data), axis=1) == 0]
    data = min_max_scaler(data)
    return data

def build_dataset(data,sequence_length):
    dataX=[]
    dataY=[]
    x=data
    y=data[:,[-1]]        # weighted price
    len_data = 0
    try:
        for i in range(0,len(data),sequence_length):
            dx=x[i:i+sequence_length]
            dy=y[i+sequence_length]
            dataX.append(dx)
            dataY.append(dy)
            len_data+=1
    except IndexError:
        return dataX,dataY,len_data
    return dataX,dataY,len_data


class Model:
    def __init__(self,n_inputs, n_sequences, n_hiddens, n_outputs, hidden_layer_cnt,istraining, file_name, model_name):
        self.n_inputs = n_inputs
        self.n_sequences = n_sequences
        self.n_hiddens = n_hiddens
        self.n_outputs = n_outputs
        self.hidden_layer_cnt = hidden_layer_cnt  # 5
        self.file_name = file_name
        self.model_name = model_name
        self.istraining=istraining
        self._build_net()


    def _build_net(self):
        with tf.name_scope('multi_lstm_layer') as scope:
            cell=tf.contrib.rnn.BasicLSTMCell(num_units=self.n_hiddens,activation=tf.tanh)
            #if self.istraining==True:
            #    cell=tf.contrib.rnn.DropoutWrapper(cell,input_keep_prob=0.9)
            multi_layer_cell=tf.contrib.rnn.MultiRNNCell([cell]*hidden_layer_cnt)
            #initial_state=cell.zero_state(dtype=tf.float32,batch_size=batch_size)
            output,_state=tf.nn.dynamic_rnn(multi_layer_cell,X,dtype=tf.float32)  #,initial_state=initial_state,dtype=tf.float32)
            #self.y_pred = tf.contrib.layers.fully_connected(output[:,-1],self.n_outputs,activation_fn=None)


            self.W_rnn = tf.trainable_variables()[0]
            self.W = tf.get_variable(name='W',shape=[7,1], dtype=tf.float32,
                                      initializer=tf.contrib.layers.variance_scaling_initializer())
            self.b = tf.Variable(tf.constant(value=0.001, shape=[1], name='b'))
            self.y_pred = tf.matmul(output[:,-1], self.W) + self.b

        #for tf_var in tf.trainable_variables():
        #    if 'kernel' in tf_var:
        #       self.W_rnn=tf_var

        self.loss=tf.reduce_sum(tf.square(self.y_pred-Y))
        self.regularizer = tf.nn.l2_loss(self.W)+tf.nn.l2_loss(self.W_rnn)
        self.cost = tf.reduce_mean(self.loss + 0.01 * self.regularizer)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        self.train=optimizer.minimize(self.cost)



file_names=['bitstampUSD_1-min_data_2012-01-01_to_2017-05-31.csv']

# parameters
n_sequence=60
data_dim=7
n_hiddens=7
n_output=1
hidden_layer_cnt=5
learning_rate=0.01
istraining=True

# 데이터셋
data=read_data(file_names[0])
dataX,dataY,len_data=build_dataset(data,sequence_length=n_sequence)

trainlim=int(0.8*len_data)
trainX,trainY=np.array(dataX[0:trainlim]),np.array(dataY[0:trainlim])
testX,testY  =np.array(dataX[trainlim:-1]),np.array(dataY[trainlim:-1])

X=tf.placeholder(tf.float32,[None,n_sequence,data_dim])
Y=tf.placeholder(tf.float32,[None,1])
loss=tf.placeholder(tf.float32,[None,1])

batch_size=2000
iteration=100   #trainlim//batch_size
num_iter=100

model=Model(batch_size,n_sequence,n_hiddens,n_output,hidden_layer_cnt,istraining,file_name=file_names[0],model_name='model1')
#iteration=trainlim//batch_size



# RMSE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
loss_test = tf.reduce_mean(tf.square(targets - predictions))

test_y=[]
test_pred=[]

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('learning start!')
    # Training step
    for e in range(num_iter):
        iteration_loss=[]
        for i in range(0,trainlim,batch_size):
        #for i in range(iteration):
            batch_X, batch_Y = np.array(trainX[i:i+batch_size]), np.array(trainY[i:i+batch_size])
            _, step_loss = sess.run([model.train, model.loss], feed_dict={X: batch_X, Y: batch_Y})
            #_, step_loss = sess.run([model.train, model.loss], feed_dict={X: trainX, Y: trainY})
            iteration_loss.append(step_loss)
        #_, epoch_loss = sess.run([model.train, model.loss], feed_dict={X:trainX, Y:trainY})
        print("[epoch {}] loss: {}".format((e + 1), np.average(iteration_loss)))
        #print("[iteration {}] loss: {}".format((e+1)*iteration,np.average(iteration_loss)))
    print('learning end!')

    # Test step
    #model.istraining=False
    test_predict = sess.run([model.y_pred], feed_dict={X: testX})
    test_predict = np.reshape(test_predict,[-1,1])
    loss_val = sess.run(loss_test, feed_dict={targets: testY, predictions: test_predict})
    print("test loss: {}".format(loss_val))

    test_y.append(testY)
    test_pred.append(test_predict)



# Plot predictions
plt.plot(np.array(test_y).reshape(-1,1),color='red')
plt.plot(np.array(test_pred).reshape(-1,1),color='blue')
plt.xlabel("Time Period")
plt.ylabel("Stock Price")
plt.show()