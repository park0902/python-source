import numpy as np
import tensorflow as tf
import librosa
import os


def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

# 텐서플로우 모델 생성
n_dim = 193
n_classes = 10
n_hidden_units_one = 300
n_hidden_units_two = 200
n_hidden_units_three = 100
sd = 1 / np.sqrt(n_dim)

X = tf.placeholder(tf.float32,[None,n_dim])
Y = tf.placeholder(tf.float32,[None,n_classes])

W_1 = tf.Variable(tf.random_normal([n_dim, n_hidden_units_one], mean=0, stddev=sd), name='W1')
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean=0, stddev=sd), name="b1")
h_1 = tf.nn.sigmoid(tf.matmul(X, W_1) + b_1)

W_2 = tf.Variable(tf.random_normal([n_hidden_units_one, n_hidden_units_two], mean=0, stddev=sd), name="W2")
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean=0, stddev=sd), name="b2")
h_2 = tf.nn.tanh(tf.matmul(h_1, W_2) + b_2)

W_3 = tf.Variable(tf.random_normal([n_hidden_units_two, n_hidden_units_three], mean=0, stddev=sd), name="W3")
b_3 = tf.Variable(tf.random_normal([n_hidden_units_three], mean=0, stddev=sd), name="b3")
h_3 = tf.nn.sigmoid(tf.matmul(h_2, W_3) + b_3)

# W = tf.Variable(tf.random_normal([n_hidden_units_three, n_classes], mean=0, stddev=sd), name="W")
# b = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=sd), name="b")
# y_ = tf.nn.softmax(tf.matmul(h_3, W) + b)

W = tf.Variable(tf.random_normal([n_hidden_units_three, n_classes], mean=0, stddev=sd), name="W")
b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd), name="b")
z = tf.matmul(h_3, W) + b
y_sigmoid = tf.nn.sigmoid(z)
y_ = tf.nn.softmax(z)




init = tf.global_variables_initializer()


# mfccs, chroma, mel, contrast, tonnetz = extract_feature('D:\park\\002-UrbanSoundData\\audio\\fold1\\21684-9-0-12.wav')
# mfccs, chroma, mel, contrast, tonnetz = extract_feature('D:\park\\002-UrbanSoundData\\audio\\fold1\\21684-9-0-25.wav')

# mfccs, chroma, mel, contrast, tonnetz = extract_feature('D:\park\\002-UrbanSoundData\\audio\\fold1\\40722-8-0-3.wav')
mfccs, chroma, mel, contrast, tonnetz = extract_feature('D:\park\\002-UrbanSoundData\\audio\\fold1\\179867-1-0-0.wav')

# mfccs, chroma, mel, contrast, tonnetz = extract_feature('D:\\park\\music\\Shock2.WAV')
x_data = np.hstack([mfccs, chroma, mel, contrast, tonnetz])

label = {0:'air_conditioner', 1:'car_horn', 2:'children_playing', 3:'dog_bark', 4:'drilling', 5:'engine_idling',
         6:'gun_shot', 7:'jackhammer', 8:'siren', 9:'street_music'}

with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    # saver.restore(sess,'D:\park\model_321\\model_321.ckpt')
    # saver.restore(sess,'D:\park\model_321_he_sce\\model_321.ckpt')
    saver.restore(sess, 'D:\park\model_321_he_sce_re\\model_321.ckpt')
    pre = sess.run(y_, feed_dict={X: x_data.reshape(1,-1)})
    # pree = sess.run(Y, feed_dict={X: x_data.reshape(1,-1)})
    print(pre)
    print(label[np.argmax(pre)])



# with tf.Session() as sess:
#     sess.run(init)
#     saver = tf.train.Saver()
#     # saver.restore(sess,'D:\park\model_321\\model_321.ckpt')
#     saver.restore(sess, 'D:\park\model_321_he_sce_re\\model_321.ckpt')
#     y_hat, sigmoid = sess.run([y_, y_sigmoid], feed_dict={X: x_data.reshape(1,-1)})
#     # pree = sess.run(Y, feed_dict={X: x_data.reshape(1,-1)})
#     print(y_hat)
#     print(sigmoid)
#     index = np.argmax(y_hat)
#     print(index)
