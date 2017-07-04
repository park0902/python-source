'''
--------------------------------------------------------------------------------------
- 단순한 계층 구현하기

    곱셈노드 : MultiLayer
    뎃셈노드 : AddLayer
--------------------------------------------------------------------------------------
'''


'''
--------------------------------------------------------------------------------------
- 곱셈 계층

    모든 계층은 forward() 와 backward() 라는 공통의 메서드(인터페이스) 를 갖도록 구현
    
    forward() : 순전파      backward() : 역전파
--------------------------------------------------------------------------------------
'''

class MultiLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        # x 와 y를 바꾼다
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


apple = 100
apple_num = 2
tax = 1.1

# 계층들
mul_apple_layer = MultiLayer()
mul_tax_layer = MultiLayer()

# 순전파
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print(price)    # 220


# 역전파
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(dapple, dapple_num, dtax)     # 2.2   110   200



'''
--------------------------------------------------------------------------------------
- 덧셈 계층

    모든 계층은 forward() 와 backward() 라는 공통의 메서드(인터페이스) 를 갖도록 구현

    forward() : 순전파      backward() : 역전파
--------------------------------------------------------------------------------------
'''

class MultiLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        # x 와 y를 바꾼다
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y

        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# 계층들
mul_apple_layer = MultiLayer()
mul_orange_layer = MultiLayer()
add_apple_layer = AddLayer()
mul_tax_layer = MultiLayer()

# 순전파
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_apple_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)

# 역전파
dprice = 1
dall_price, dtax = add_apple_layer.backward(dprice)
dapple_price, dorange_price = add_apple_layer.backward(dall_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(price)    # 715.0000000000001
print(dapple_num, dapple, dorange, dorange_num, dtax)   # 100 2 3 150 1



'''
--------------------------------------------------------------------------------------
- ReLU 계층

    활성화 함수로 사용되는 ReLU 수식
    
            x   (x >  0)
        y = 
            0   (x <= 0)
            
            
        ay     1   (x >  0)
       ---- = 
        ax     0   (x <= 0)
--------------------------------------------------------------------------------------
'''

import numpy as np

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

x = np.array([[1.0, -0.5], [-2.0, 3.0]])
mask = (x <= 0)

print(x)
print(mask)



'''
--------------------------------------------------------------------------------------
- sigmoid 계층

    모든 계층은 forward() 와 backward() 라는 공통의 메서드(인터페이스) 를 갖도록 구현

    forward() : 순전파      backward() : 역전파
--------------------------------------------------------------------------------------
'''

import numpy as np

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx



'''
--------------------------------------------------------------------------------------
- 배치용 Affine 계층

    신경망의 순전파 때 수행하는 행렬의 내적은 기하학에서는 어파인 변환이라고 한다!!
    
    어파인 변환을 수행하는 처리를 Affine 계층이라는 이름으로 구현
--------------------------------------------------------------------------------------
'''

import numpy as np

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.W.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx



'''
--------------------------------------------------------------------------------------
- Softmax-with-Loss 계층

    마지막 출력층에서 사용하는 소프트맥스 함수
    소프트맥스 함수는 입력 값을 정규화하여 출력!
    
    신경망에서 수행하는 작업은 학습과 추론!
    
        추론할 때는 Softmax 계층 사용 X
        
        신경망은 추론할 때는 마지막 Affine 계층의 출력을 인식결과로 이용
        신경망에서 정규화하지 않은 출력결과를 점수(score) 라고 한다!!
        
        즉, 신경망 추론애소 답을 하나만 내는 경우에는 가장 높은 점수만 알면 되니 Softmax 계층은 필요X
        
    
        학습할 때는 Softmax 계층 필요!!
--------------------------------------------------------------------------------------
'''

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None    # 손실
        self.y = None       # softmax의 출력
        self.x = None       # 정답 레이블(원-핫 벡터)


    def cross_entropy_error(self, y, t):
        if y.ndim == 1:
            self.t = t.reshape(1, t.size)
            self.y = y.reshape(1, y.size)

        # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
        if t.size == y.size:
            self.t = t.argmax(axis=1)

        batch_size = y.shape[0]


        return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

    def forward(self, x,t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss


