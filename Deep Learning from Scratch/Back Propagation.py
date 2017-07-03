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




