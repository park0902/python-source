########################### 사용 모듈 ############################
#	 numpy
#	 matplotlib
#	 pandas
#	 statsmodels
###################################################################


########################### 단순 선형 회귀 numpy 모듈 사용 #####################################


## 1. 최소제곱법 메소드인 .lstsq 와 선형대수 메소드인 .linalg 를 사용
import numpy as np
import matplotlib.pyplot as plt

##임산부의 에스트리올 수치(x)가 출생시 체중(y)에 미치는 영향?
x = np.array([7,9,9,12,14,16,16,14,16,16,17,19,21,24,15,16,17,25,27,15,15,15,16,19,18,17,18,20,22,25,24])
y = np.array([25,25,25,27,27,27,24,30,30,31,30,31,30,28,32,32,32,32,34,34,34,35,35,34,35,36,37,38,40,39,43])
A = np.vstack([x, np.ones(len(x))]).T
print(x)
#a=알파, b=베타
a, b = np.linalg.lstsq(A, y)[0] #Return the least-squares solution to a linear matrix equation.

#산점도
plt.plot(x, y, 'o', label='data', markersize=8)
plt.hold(True)
plt.plot(x, a*x + b, 'r', label='Fitted line')
plt.hold(False)
plt.legend()
plt.show()
#회귀식
print('출생시 체중 = ',a,' * 에스트리올 + ',b)
# 출생시 체중 =  0.60819047619  * 에스트리올 +  21.5234285714

#simple linear regression
import numpy as np

x = np.array([7,9,9,12,14,16,16,14,16,16,17,19,21,24,15,16,17,25,27,15,15,15,16,19,18,17,18,20,22,25,24])
y = np.array([25,25,25,27,27,27,24,30,30,31,30,31,30,28,32,32,32,32,34,34,34,35,35,34,35,36,37,38,40,39,43])


def least_squares_fit(x,y):
    beta = np.corrcoef(x,y)[0][1] * np.std(y) / np.std(x)
    alpha = np.mean(y) - beta * np.mean(x)
    return alpha, beta

print ( least_squares_fit(x,y))
# 출생시 체중 = 0.60819047619047628 * 에스트리올 + 21.523428571428568

########################## 다중 선형 회귀 #####################################
# mutiple linear regression
# data : http://cafe.naver.com/office2080/1271
# y : 야구선수 연봉
# x1: 부양가족 x2:년수 x3: 승률
# beta = (x'x)-1x'y
# 베타 계수 구하는 함수
import numpy.linalg as lin
import numpy as np

y = np.array ([ [100],[150],[200],[250],[300],[350],[400],[450],[500],[550] ])
x= np.array( [[1,1,1,20],[1,1,2,25],[1,1,3,30],[1,4,4,20],[1,6,2,25],[1,2,1,20],[1,4,2,25],[1,2,3,55],[1,2,4,60],[1,4,2,60] ])

def least_squares_fit2(x,y):
	temp = lin.inv(np.dot(x.T,x))
	temp2 =np.dot( temp, x.T, )
	beta = np.dot(temp2, y)
	return beta

print ( least_squares_fit2(x,y))

# [[ 13.27414438] #b0(절편)
#  [ 35.06626867] #b1
#  [-14.93054227] #b2
#  [  7.43765387]] #b3

############################ 단순 선형 회귀 시각화 ######################################

import pandas as pd
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
import numpy as np

df = pd.read_csv('D:\data\sports.csv', index_col=0)

fig, axs = plt.subplots(1, 3, 'row')

df.plot(kind='scatter', x='sports', y='acceptance', ax=axs[0], figsize=(16, 8))
df.plot(kind='scatter', x='music', y='acceptance', ax=axs[1])
df.plot(kind='scatter', x='academic', y='acceptance', ax=axs[2])

# create a fitted model in one line
lm = smf.ols(formula='acceptance ~ music', data=df).fit()

X_new = pd.DataFrame({'music': [df.music.min(), df.music.max()]})
preds = lm.predict(X_new)

df.plot(kind='scatter', x='music', y='acceptance', figsize=(12,12), s=50)

plt.title("Linear Regression - Fitting Music vs Acceptance Rate", fontsize=20)
plt.xlabel("Music", fontsize=16)
plt.ylabel("Acceptance", fontsize=16)

plt.plot(X_new, preds, c='red', linewidth=2)
#
# #      academic  sports       music  acceptance
# # 1       230.1    37.8   62.909091   81.851852
# # 2        44.5    39.3   41.000000   38.518519
# # 3        17.2    45.9   63.000000   34.444444
# # 4       151.5    41.3   68.518519   68.518519
# # 5       180.8    10.8   53.090909   47.777778
# # 6         8.7    48.9   68.181818   26.666667

musicpd = list(df['music'])
acceptancepd = list(df['acceptance'])

musicx = np.array(musicpd)
acceptancey = np.array(acceptancepd)

A = np.vstack([musicx, np.ones(len(musicx))]).T

#a=알파, b=베타
a, b = np.linalg.lstsq(A, acceptancey)[0] #Return the least-squares solution to a linear matrix equation.

# #산점도
# plt.plot(musicx, acceptancey, 'o', label='data', markersize=8)
# plt.plot(musicx, a*musicx + b, 'r', label='Fitted line')
# plt.legend()
# plt.show()
#회귀식
print('대학교 합격률 = ',a,' * 음악성적 + ',b)

# 대학교 합격률 =  0.511925556662  * 음악성적 +  31.2988713351