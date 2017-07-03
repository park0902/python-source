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

print(dapple, dapple_num, dtax)     # 2.2   110     200