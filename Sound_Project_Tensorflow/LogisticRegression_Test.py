import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
"""
훈련 집합
1 ~ 4	긁힘	[1,0]
5 ~ 7	충격	[0,1]


테스트 집합
8 ~ 10	긁힘	[1,0]
11 ~ 12	충격	[0,1]

"""


import glob
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from sklearn import linear_model

# 데이터 읽어오기. wav 파일이 여려 개이므로 이를 리스트에 담는다.
y = []
sr = []
for i in range(10):
    file = 'D:\park\Logistic_music\\'+str(i + 1) + '.WAV'

    yt, srt = librosa.load(file)
    y.append(yt)
    sr.append(srt)

# mfcc 변환. 오디오 신호를 mfcc로 바꾼다.
mfcc = []

# n_mfcc를 784가 아닌 25로 한다.
for i in range(10):
    mfcc.append(librosa.feature.mfcc(y=y[i], sr=sr[i], n_mfcc=20))

# shape 확인을 위한 과정
# print("mfcc.shape : ", mfcc.shape) 를 실행하면 AttributeError: 'list' object has no attribute 'shape'
# print("mfcc len : ", len(mfcc))
# print("mfcc[0].shape : ", mfcc[0].shape)

# 200행의 입력 파일이 존재, 25열의 변수, 각 변수 하나는 31개의 수치로 구성.

# 데이터 처리를 용이하게 하기 위해 평균을 사용한다.
# 31개로 이루어진 부분을 평균을 내야 한다.

mfccMean = []
for i in range(10):
    mfccMean.append(np.mean(mfcc[i], axis = 1))

# mfccMean의 shape을 확인하기.

mfccMeanArray = np.asarray(mfccMean)
# print("mfccMeanArray shape : ", mfccMeanArray.shape)
# print("mfccMeanArray[0][0] :", mfccMeanArray[0][0])
# print(mfccMeanArray)


# train 집합과 test 집합을 만들자.
# 총 200개의 파일이 있다. 150개를 훈련 집합으로, 50개를 테스트 집합으로 정하자.
# 훈련 집합에서 1 ~ 132의 레이블링은 [1,0]이다. 133 ~ 150의 레이블링은 [0,1]이다.
# 테스트 집합에서 151 ~ 190의 레이블링은 [1,0]이다. 191 ~ 200은 [0,1]이다.
# [True, False] 형태로 logits를 구성한다.

x_train = []
y_train = []
x_test = []
y_test = []
print(len(mfccMean))


for i in range(6):
    x_train.append(mfccMean[i])
    if i == 0 or i%2 ==0:
        y_train.append(1)
    else:
        y_train.append(0)

for i in range(6, 10):
    x_test.append(mfccMean[i])
    if i%2==0:
        y_test.append(1)
    else:
        y_test.append(0)



# list를 array로
x_train_array = np.asarray(x_train)
y_train_array = np.asarray(y_train)
x_test_array = np.asarray(x_test)
y_test_array = np.asarray(y_test)

print(x_train_array)
print(x_train_array.shape)
print()
print(y_train_array)
print(y_train_array.shape)
print()
print(x_test_array)
print(x_test_array.shape)
print()
print(y_test_array)
print(y_test_array.shape)
print()
#




# re_y = []
# re_sr = []

re_yt, re_srt = librosa.load('D:\park\Logistic_music\\Scratch6.WAV')
# re_y.append(re_yt)
# re_sr.append(re_srt)

# mfcc 변환. 오디오 신호를 mfcc로 바꾼다.
# re_mfcc = []

# n_mfcc를 784가 아닌 25로 한다.

re_mfcc = librosa.feature.mfcc(y=re_yt, sr=re_srt, n_mfcc=20)

# shape 확인을 위한 과정
# print("mfcc.shape : ", mfcc.shape) 를 실행하면 AttributeError: 'list' object has no attribute 'shape'
# print("mfcc len : ", len(mfcc))
# print("mfcc[0].shape : ", mfcc[0].shape)

# 200행의 입력 파일이 존재, 25열의 변수, 각 변수 하나는 31개의 수치로 구성.

# 데이터 처리를 용이하게 하기 위해 평균을 사용한다.
# 31개로 이루어진 부분을 평균을 내야 한다.

re_mfccMean = np.mean(re_mfcc, axis = 1)
re_test_array = [1]

re_test_array = np.asarray(re_test_array)
re_mfccMean = np.asarray(re_mfccMean).reshape(1,20)

print(re_test_array.shape)
print(re_mfccMean.shape)
print(re_mfccMean)





import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


logreg = linear_model.LogisticRegression()


logreg.fit(x_train, y_train)

y_test_estimated = logreg.predict(re_mfccMean)

print('예측')
print(y_test_estimated)
print()
print('실제')
print(re_test_array)










# x_train_array = x_train_array.reshape(-1, 5, 5, 1)
# x_test_array = x_test_array.reshape(-1, 5, 5, 1)
#
#
# print("x_train_array : ", x_train_array.shape)
# print("y_train_array : ", y_train_array.shape)
# print("x_test_array : ", x_test_array.shape)
# print("y_test_array : ", y_test_array.shape)
#
#
#
# # import some data to play with
# iris = datasets.load_iris()
# X = iris.data[:, :2]  # we only take the first two features.
# Y = iris.target
#
# h = .02  # step size in the mesh
#
# logreg = linear_model.LogisticRegression(C=1e5)
#
# # we create an instance of Neighbours Classifier and fit the data.
# logreg.fit(X, Y)
#
# # Plot the decision boundary. For that, we will assign a color to each
# # point in the mesh [x_min, x_max]x[y_min, y_max].
# x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
# y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
#
# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure(1, figsize=(4, 3))
# plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
#
# # Plot also the training points
# plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')
#
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.xticks(())
# plt.yticks(())
#
# plt.show()