'''
--------------------------------------------------------------------------------------
- 학습

    훈련 데이터로부터 가중치 매개변수의 최적값을 자동으로 획득하는 것

    
- 손실함수

    신경망이 학습할 수 있도록 해주는 지표(평균 제곱 오차, 교차 엔트로피 오차 사용)
    
=> 손실함수의 결과값을 가장 작게 만드는 가중치 매개변수를 찾는 것이 목표!!!


- 오버피팅

    한 데이터셋에만 지나치게 최적화된 상태
--------------------------------------------------------------------------------------
'''

'''
--------------------------------------------------------------------------------------
- 평균 제곱 오차

          1
    E =  --- 시그마 (yk - tk)의 제곱
          2
          
        
        yk : 신경망의 출력(신경망이 추정한 값)
        tk : 정답 레이블
        k : 데이터의 차원 수
--------------------------------------------------------------------------------------
'''

import numpy as np

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t) ** 2)

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
y1 = [0.1 ,0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]

print(mean_squared_error(np.array(y), np.array(t)))     # 정답도 2이고, 신경망의 출력도 2   0.0975....
print(mean_squared_error(np.array(y1), np.array(t)))    # 정답은 2이고, 신경망의 출력은 7   0.5975.....

# 평균 제곱 오차를 기준으로 오차가 더 작은 값이 정답에 가까울 것으로 판단!!!



'''
--------------------------------------------------------------------------------------
- 교차 엔트로피 오차

          
    E =  - 시그마 (tk * log yk)
          


        yk : 신경망의 출력(신경망이 추정한 값)
        tk : 정답 레이블
        k : 데이터의 차원 수
--------------------------------------------------------------------------------------
'''

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
y1 = [0.1 ,0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]

print(cross_entropy_error(np.array(y), np.array(t)))    # 정답일때 출력이 0.6인경우 교차 엔트로피 오차는 약 0.51
print(cross_entropy_error(np.array(y1), np.array(t)))   # 정답일때 출력이 0.1인경우 교차 엔트로피 오차는 무려 2.3

# 오차 값이 더 작은 첫 번째 추정이 정답일 가능성이 높다고 판단한 것으로 앞서 평균 제곱 오차의 판단과 일치!!!




'''
--------------------------------------------------------------------------------------
- 미니배치 학습

    가령 60000장의 훈련 데이터 중에서 100장을 무작위로 뽑아 그 100장만을 사용하여 학습하는 것
    이러한 학습 방법을 미니배치 학습이라고 한다!!!


    기계학습 문제
    
        훈련 데이터에 대한 손실 함수의 값을 구하고, 그 값을 최대한 줄여주는 매개변수를 찾아낸다!
        
        모든 훈련 데이터를 대상으로 손실 함수의 값을 구해야 한다!!!
        
    
    교차 엔트로피 오차
    
                  1  
        E  =  - ----- 시그마(n) 시그마(k) * tnk * log ynk
                  N
                  
                  
        데이터가 N개라면   tnk : n 번째 데이터의 k번째 값(정답 레이블)
                        ynk : 신경망의 출력
--------------------------------------------------------------------------------------
'''

import  sys, os
sys.path.append(os.pardir)
import numpy as np
from MNIST import  load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)    # (60000, 784)
print(t_train.shape)    # (60000, 10)

# 훈련데이터는 60000개, 입력데이터는 784열(원래 28 X 28)인 이미지 데이터
# 정답 레이블은 10줄짜리 데이터


# 훈련데이터에서 무작위로 10장만 빼내기
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

print(batch_mask)




'''
--------------------------------------------------------------------------------------
- (배치용) 교차 엔트로피 오차 구현
--------------------------------------------------------------------------------------
'''

import numpy as np

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y)) / batch_size

'''
=> y : 신경망의 출력
   t : 정답 레이블
   
=> y 가 1차원이라면, 즉 데이터 하나당 교차 엔트로피 오차를 구하는 경우는 reshape 함수로 데이터의 형상을 바꿔준다!!

=> 배치의 크기로 나눠 정규화하고 이미지 1장당 평균의 교차 엔트로피 오차 계산!!
--------------------------------------------------------------------------------------
'''



'''
--------------------------------------------------------------------------------------
- (배치용) 교차 엔트로피 오차 구현 (정답 레이블이 원-핫 인코딩이 아니라 숫자 레이블로 주어졌을때)
--------------------------------------------------------------------------------------
'''

import numpy as np

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

'''
=> batch_size 가 5이면 np.arrange(batch_size)는 [0, 1, 2, 3, 4] 라는 넘파이 배열 생성

=> t 는 숫자레이블이므로 t = [2, 7, 0, 9, 4] 와 같이 저장되어 있으므로
   y[np.arange(batch_size), t] 는 [y[0,2], y[1,7], y[2,0], y[3,9], y[4,4]] 인 넘파이 배열 생성   
--------------------------------------------------------------------------------------
'''



'''
--------------------------------------------------------------------------------------
- 왜 손실 함수를 설정하는가?

    신경망 학습에서 최적의 매개변수(가중치와 편향)를 탐색할 때 손실 함수의 값을 가능한 작게 하는 매개변수 값을 찾는다!!
    이때 매개변수의 미분(정확히는 기울기)을 계산하고, 그 미분 값을 단서로 매개변수의 값을 서서히 갱신하는 과정 반복!!
    
    신경망을 학습할 때 정확도를 지표로 삼아서는 안 된다!!
    정확도를 지표로 하면 매개변수의 미분이 대부분의 장소에서 0 이 되기 때문이다!!
    
    계단 함수는 대부분의 장소에서 기울기가 0이지만, 시그모이드 함수의 기울기(접선)는 0이 아니다 
--------------------------------------------------------------------------------------
'''



'''
--------------------------------------------------------------------------------------
- 수치 미분

    아주 작은 차분으로 미분하는 것을 수치 미분
    
    해석적 미분은 오차를 포함하지 않은 진정한 미분 값을 구해준다!!
 
       df(x)                  f(x+h) - f(x)
     --------  = lim(h->0) --------------------
        dx                           h
        
        
     h : 10 의 -4 정도의 값을 사용하면 좋은 결과를 얻는다고 알려져 있다
     
     수치 미분에는 오차가 포함된다!!
     이 오차를 줄이기 위해 (x + h) 와 (x - h) 일 때의 함수 f의 차분을 계산하는 방법을 쓰기도 한다!!
     이 차분은 x를 중심으로 그 전후의 차분을 계산하다는 의미에서 중심 차분 혹은 중앙 차분 이라 한다!!
     (한편, (x + h) 와 x 의 차분은 전방 차분 이라 한다)
--------------------------------------------------------------------------------------
'''

def numerical_diff(f, x):
    h = 1e-4    # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)




'''
--------------------------------------------------------------------------------------
- 수치 미분의 예

        y = 0.01 x2 + 0.1 x
--------------------------------------------------------------------------------------
'''

import numpy as np
import matplotlib.pyplot as plt


def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x


def numerical_diff(f, x):
    h = 1e-4    # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)


x = np.arange(0.0, 20.0, 0.1)   # 0 에서 20 까지 0.1 간격의 배열 x 생성
y = function_1(x)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()


# x = 5 일 때와 10 일 때 위 함수의 미분 계산
print(numerical_diff(function_1, 5))
print(numerical_diff(function_1, 10))




'''
--------------------------------------------------------------------------------------
- 편미분

    변수가 여럿인 함수에 대한 미분

    f(x0, x1) = x0^2 + x1^2
    
    어느 변수에 대한 미분이냐. 즉. x0 와 x1 중 어느 변수에 대한 미분이냐를 구분해야 한다!!
--------------------------------------------------------------------------------------
'''

import numpy as np

def function_2(x):
    return np.sum(x ** 2)
    # 또는 return x[0] ** 2 + x[1] ** 2


#                                      af
# x0 = 3, x1 = 4 일때 x0 에 대한 편미분 ------ 구하기
#                                      ax0

def numerical_diff(f, x):
    h = 1e-4    # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)


def function_tmp1(x0):
    return x0*x0 + 4.0**2.0

print(numerical_diff(function_tmp1, 3.0))


#                                      af
# x0 = 3, x1 = 4 일때 x1 에 대한 편미분 ------ 구하기
#                                      ax1

def numerical_diff(f, x):
    h = 1e-4    # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)


def function_tmp2(x1):
    return 3.0**2 + x1*x1

print(numerical_diff(function_tmp2, 4.0))

'''
=> 편미분은 변수가 하나인 미분과 마찬가지로 특정 장소의 기울기를 구한다!!
   단 여러 변수 중 목표 변수 하나에 초점을 맞추고 다른 변수는 값을 고정한다!!
--------------------------------------------------------------------------------------
'''




'''
--------------------------------------------------------------------------------------
- 기울기

    모든 변수의 편미분을 벡터로 정리한 것
--------------------------------------------------------------------------------------
'''

import numpy as np

def numerical_gradient(f, x):
    h = 1e-4    # 0.0001
    grad = np.zeros_like(x)     # x 와 형상이 같고 그 원소가 모드 0인 배열 생성

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val    # 값 복원

    return grad

print(numerical_gradient(function_2, np.array([3.0, 4.0])))
print(numerical_gradient(function_2, np.array([0.0, 2.0])))
print(numerical_gradient(function_2, np.array([3.0, 0.0])))



# 기울기의 결과에 마이너스를 붙인 벡터 그래프
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


def _numerical_gradient_no_batch(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # x와 형상이 같은 배열을 생성

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h) 계산
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 값 복원

    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)

        return grad


def function_2(x):
    if x.ndim == 1:
        return np.sum(x ** 2)
    else:
        return np.sum(x ** 2, axis=1)


def tangent_line(f, x):
    d = numerical_gradient(f, x)
    print(d)
    y = f(x) - d * x
    return lambda t: d * t + y


if __name__ == '__main__':
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)

    X = X.flatten()
    Y = Y.flatten()

    grad = numerical_gradient(function_2, np.array([X, Y]))

    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1], angles="xy", color="#666666")  # ,headwidth=10,scale=40,color="#444444")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()





