# # 곱셈계층 class 구현
# class MulLayer:
#     def __init__(self):
#         self.x = None
#         self.y = None
#
#     def forward(self, x, y):
#         self.x = x
#         self.y = y
#         out = x * y
#
#         return out
#
#     def backward(self, dout):
#         dx = dout * self.y
#         dy = dout * self.x
#
#         return dx, dy
#
#
#
#
# # 곱셈 클래스를 객체화 시켜서 사과가격 구하기(순전파)
# class MulLayer:
#     def __init__(self):
#         self.x = None
#         self.y = None
#
#     def forward(self, x, y):
#         self.x = x
#         self.y = y
#         out = x * y
#
#         return out
#
#     def backward(self, dout):
#         dx = dout * self.y
#         dy = dout * self.x
#
#         return dx, dy
#
# apple = 200
# apple_num = 5
# tax = 1.2
#
# mul_apple_layer = MulLayer()
# mul_apple_tax_layer = MulLayer()
#
# apple_price = mul_apple_layer.forward(apple, apple_num)
# price = mul_apple_tax_layer.forward(apple_price, tax)
#
# print(price)
#
#
#
#
# # 덧셈계층 class 구현
# class AddLayer:
#     def __init__(self):
#         pass
#
#     def foward(self, x, y):
#         out = x + y
#         return out
#
#     def backward(self,dout):
#         dx = dout * 1
#         dy = dout * 1
#
#         return dx, dy
#
#
#
#
# # 사과 2개와 귤 5개를 구입하면 총 가격이 얼마인지 구하시오
# class MulLayer:
#     def __init__(self):
#         self.x = None
#         self.y = None
#
#     def forward(self, x, y):
#         self.x = x
#         self.y = y
#         out = x * y
#
#         return out
#
#     def backward(self, dout):
#         dx = dout * self.y
#         dy = dout * self.x
#
#         return dx, dy
#
#
# class AddLayer:
#     def __init__(self):
#         pass
#
#     def foward(self, x, y):
#         out = x + y
#         return out
#
#     def backward(self, dout):
#         dx = dout
#         dy = dout
#
#         return dx, dy
#
#
#
#
# # 사과 2개와 귤 5개를 구입하면 총 가격이 얼마인지 구하시오
# class MulLayer:
#     def __init__(self):
#         self.x = None
#         self.y = None
#
#     def forward(self, x, y):
#         self.x = x
#         self.y = y
#         out = x * y
#
#         return out
#
#     def backward(self, dout):
#         dx = dout * self.y
#         dy = dout * self.x
#
#         return dx, dy
#
#
# class AddLayer:
#     def __init__(self):
#         pass
#
#     def foward(self, x, y):
#         out = x + y
#         return out
#
#     def backward(self, dout):
#         dx = dout
#         dy = dout
#
#         return dx, dy
#
# apple = 200
# apple_num = 2
# orange = 300
# orange_num = 5
# tax = 1.5
#
# mul_apple_layer = MulLayer()
# mul_orange_layer = MulLayer()
# mul_tax_layer = MulLayer()
# add_apple_orange_layer = AddLayer()
#
# apple_price = mul_apple_layer.forward(apple, apple_num)
# orange_price = mul_orange_layer.forward(orange, orange_num)
# apple_orange_price = add_apple_orange_layer.foward(apple_price, orange_price)
# all_price = mul_tax_layer.forward(apple_orange_price, tax)
#
# print(all_price)
#




# 역전파 구현
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
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
        dx = dout
        dy = dout

        return dx, dy

apple = 200
apple_num = 2
orange = 300
orange_num = 5
tax = 1.5
dprice = 1

mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()


apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
apple_orange_price = add_apple_orange_layer.forward(apple_price, orange_price)
all_price = mul_tax_layer.forward(apple_orange_price, tax)



dall_price, dtax = mul_tax_layer.backward(1)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(dapple_num, dapple, dorange, dorange_num, dtax)




# 순전파 구현
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
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
        dx = dout
        dy = dout

        return dx, dy

apple, apple_num = 100, 4
orange, orange_num = 200, 3
pear, pear_num = 300, 2
tax = 1.3

mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
mul_pear_layer = MulLayer()
add_apple_orange_peer_layer = AddLayer()
mul_tax_layer = MulLayer()

a = mul_apple_layer.forward(apple, apple_num)
b = mul_orange_layer.forward(orange, orange_num)
c = mul_pear_layer.forward(pear, pear_num)

d = add_apple_orange_peer_layer.forward(a,b)
e = add_apple_orange_peer_layer.forward(c,d)

all_price = mul_tax_layer.forward(e, tax)

print(all_price)



# 역전파 구현
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
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
        dx = dout
        dy = dout

        return dx, dy

apple, apple_num = 100, 4
orange, orange_num = 200, 3
pear, pear_num = 300, 2
tax = 1.3

mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
mul_pear_layer = MulLayer()
add_all_layer = AddLayer()
mul_tax_layer = MulLayer()

a = mul_apple_layer.forward(apple, apple_num)
b = mul_orange_layer.forward(orange, orange_num)
c = mul_pear_layer.forward(pear, pear_num)

d = add_all_layer.forward(a,b)
e = add_all_layer.forward(c,d)

all_price = mul_tax_layer.forward(e, tax)

dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)



# numpy에서의 copy
import numpy as np
import copy

x = np.array([[1.0, -0.5], [-2.0, 3.0]])
print(x)

mask = (x <= 0)
print(mask)

out = x.copy()
print(out)

out[mask] = 0   # 0 이하인것은 다 0 으로 변경해주는 작업
print(out)

print(x)        # copy 가 되었기 때문에 별도의 객체인 out 이 생성
                # x 객체와는 별도로 out 을 따로 변경한 것이다


# ReLu class
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




# x 변수를 생성하고 x 를 ReLu 객체에 forward 함수에 값을 넣으면 무엇이 출력되는지 확인
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

relu = Relu()

x = np.array([[1.0, -0.5], [-2.0, 3.0]])

print(relu.forward(x))



# Sigmoid class
import numpy as np

class sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx





#
import numpy as np

x = np.array([5,6])
w = np.array([[2,4,4,],[6,3,5]])

print(np.dot(x,w))




# numpy 를 이용해 행렬 내적 구하기
import numpy as np

x = np.array([[1,2,3], [4,5,6]])
y = np.array([[1,2], [3,4], [5,6]])

print(np.dot(x,y))


import numpy as np

x = np.array([[1,2], [3,4], [5,6]])
y = np.array([7,8])
y2 = np.array([7,8], ndmin=2)
w = np.dot(x,y)

print(x.shape)
print(y.shape)
print(y2.shape)
print(w.shape)
print(w)


import numpy as np

x = np.array([[1,2],[3,4],[5,6]])
w = np.array([[1,2,3,4],[5,6,7,8]])

print(np.dot(x,w))




# 신경망을 행렬의 내적으로 구현해서 출력값 y 출력
import numpy as np

x = np.array([1,2])
w = np.array([[1,3,5], [2,4,6]])

print(np.dot(x,w))




# 순전파를 구하는 함수 forward 이름으로 생성
import numpy as np

def forward(x,w,b):
    out = np.dot(x,w) + b
    return out

x = np.array([1,2])
w = np.array([[1,3,5], [2,4,6]])
b = np.array([1,2,3])

print(forward(x,w,b))



# 역전파를 구하는 함수 backward 이름으로 생성
import numpy as np

def backward(x,w,out):
    dx = np.dot(out, w.T)
    dw = np.dot(x.T, out)
    db = np.sum(out, axis=0)

    return dx, dw, db

x = np.array([1,2], ndmin=2)
w = np.array([[1,3,5], [2,4,6]])
out = np.array([6,13,20], ndmin=2)
b = np.array([1,2,3])

print(backward(x,w,out))



# Affine class 생성
import numpy as np

class Affine:
    def __init__(self,w,b):
        self.w = w
        self.b = b


    def forward(self,x):
        return np.dot(x, self.w) + self.b



    def backward(self, x, out):
        self.dx = np.dot(out, self.w.T)
        self.dw = np.dot(x.T, out)
        self.db = np.sum(out, axis=0)

        return self.dx, self.dw, self.db

x = np.array([1,2], ndmin=2)
b = np.array([1,2,3])
w = np.array([[1,3,5], [2,4,6]])

affine1 = Affine(w,b)

print(affine1.forward(x))
print(affine1.backward(x,out))



# 2층 신경망의 순전파를 Affine 클래스를 사용해서 출력
import numpy as np

class Affine:
    def __init__(self,w,b):
        self.w = w
        self.b = b


    def forward(self,x):
        return np.dot(x, self.w) + self.b



    def backward(self, x, out):
        self.dx = np.dot(out, self.w.T)
        self.dw = np.dot(x.T, out)
        self.db = np.sum(out, axis=0)

        return self.dx, self.dw, self.db

x = np.array([1,2])
w1 = np.array([[1,3,5],[2,4,6]])
w2 = np.array([[1,4],[2,5],[3,6]])
b1 = np.array([1,2,3])
b2 = np.array([1,2])

affine1 = Affine(w1,b1)
affine12 = Affine(w2,b2)
out = affine1.forward(x)

print(affine12.forward(out))




# 2층 신경망의 역전파를 Affine 클래스를 사용해서 출력
import numpy as np

class Affine:
    def __init__(self,w,b):
        self.w = w
        self.b = b


    def forward(self,x):
        return np.dot(x, self.w) + self.b



    def backward(self, x, out):
        self.dx = np.dot(out, self.w.T)
        self.dw = np.dot(x.T, out)
        self.db = np.sum(out, axis=0)

        return self.dx, self.dw, self.db

x = np.array([1,2], ndmin=2)
w1 = np.array([[1,3,5],[2,4,6]])
w2 = np.array([[1,4],[2,5],[3,6]])
b1 = np.array([1,2,3])
b2 = np.array([1,2])

affine1 = Affine(w1,b1)
affine2 = Affine(w2,b2)
out1 = affine1.forward(x)
out2 = affine2.forward(out1)

dx2, dw2, db2 = affine2.backward(out1, out2)
dx1, dw1, db1 = affine1.backward(x, dx2)

print(dx1, dw1, db1)




# 2층 신경망의 순전파를 구현하는데 은닉층 활성화 함수로 ReLu 함수 추가해서 구현
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


class Affine:
    def __init__(self,w,b):
        self.w = w
        self.b = b


    def forward(self,x):
        return np.dot(x, self.w) + self.b



    def backward(self, x, out):
        self.dx = np.dot(out, self.w.T)
        self.dw = np.dot(x.T, out)
        self.db = np.sum(out, axis=0)

        return self.dx, self.dw, self.db

x = np.array([1,2], ndmin=2)
w1 = np.array([[1,3,5],[2,4,6]])
w2 = np.array([[1,4],[2,5],[3,6]])
b1 = np.array([1,2,3])
b2 = np.array([1,2])

relu = Relu()
affine1 = Affine(w1,b1)
affine2 = Affine(w2,b2)

out1 = affine1.forward(x)
a_r = relu.forward(out1)
out2 = affine2.forward(out1)

print(out2)




# 2층 신경망의 역전파를 구현하는데 은닉층 활성화 함수로 ReLu 함수 추가해서 구현
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


class Affine:
    def __init__(self,w,b):
        self.w = w
        self.b = b


    def forward(self,x):
        return np.dot(x, self.w) + self.b



    def backward(self, x, out):
        self.dx = np.dot(out, self.w.T)
        self.dw = np.dot(x.T, out)
        self.db = np.sum(out, axis=0)

        return self.dx, self.dw, self.db


x   = np.array([1,2], ndmin=2)
w1   = np.array([[1,3,5],[2,4,6]])
w2   = np.array([[1,4],[2,5],[3,6]])
b1   = np.array([1,2,3])
b2   = np.array([1,2])

affine1 = Affine(w1, b1)
affine2 = Affine(w2, b2)
relu1 = Relu()

out = affine1.forward(x)
out_act = relu1.forward(out)
out2 = affine2.forward(out_act)
print(out2)

dx2, dw2, db2 = affine2.backward(out_act, out2)
dx1 = relu1.backward(dx2)
dx, dw, db = affine1.backward(x, dx1)
print('dx\n', dx, '\ndw\n', dw, '\ndb\n', db)













import tensorflow as tf


def batch_norm(self, input, shape, training, convl=True, name='BN'):
    beta = tf.Variable(tf.constant(0.0, shape=[shape]), name='beta', trainable=True)
    scale = tf.Variable(tf.constant(1.0, shape=[shape]), name='gamma', trainable=True)
    if convl:
        batch_mean, batch_var = tf.nn.moments(input, [0, 1, 2], name='moments')
    else:
        batch_mean, batch_var = tf.nn.moments(input, [0], name='moments')

    ema = tf.train.ExponentialMovingAverage(decay=0.5)


    def mean_var_with_update():
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies([update_ops]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(training,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    return tf.nn.batch_normalization(input, mean, var, beta, scale, variance_epsilon=1e-3, name=name)






def batch_norm(self, input, shape, training, convl=True, name='BN'):
    beta = tf.Variable(tf.constant(0.0, shape=[shape]), name='beta', trainable=True)
    scale = tf.Variable(tf.constant(1.0, shape=[shape]), name='gamma', trainable=True)
    if convl:
        batch_mean, batch_var = tf.nn.moments(input, [0, 1, 2], name='moments')
    else:
        batch_mean, batch_var = tf.nn.moments(input, [0], name='moments')

    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            # total_loss = control_flow_ops.with_dependencies([updates], total_loss)

        with tf.control_dependencies([update_ops]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(training,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    return tf.nn.batch_normalization(input, mean, var, beta, scale, variance_epsilon=1e-3, name=name)











#########################################################################################

def batch_norm(self,input, shape, training, convl=True, name='BN'):
    beta = tf.Variable(tf.constant(0.0, shape=[shape]), name='beta', trainable=True)
    scale = tf.Variable(tf.constant(1.0, shape=[shape]), name='gamma', trainable=True)
    if convl:
        batch_mean, batch_var = tf.nn.moments(input, [0, 1, 2], name='moments')
    else:
        batch_mean, batch_var = tf.nn.moments(input, [0], name='moments')

    def mean_var_with_update():
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        updates = tf.group(*update_ops)

        with tf.control_dependencies([updates]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = tf.cond(training,
                         mean_var_with_update,
                        lambda: (batch_mean, batch_var))

    return tf.nn.batch_normalization(input, mean, var, beta, scale, variance_epsilon=1e-3, name=name)

###################################################################################################

def batch_norm(self, input, shape, training, convl=True, name='BN'):
    beta = tf.Variable(tf.constant(0.0, shape=[shape]), name='beta', trainable=True)
    scale = tf.Variable(tf.constant(1.0, shape=[shape]), name='gamma', trainable=True)
    if convl:
        batch_mean, batch_var = tf.nn.moments(input, [0, 1, 2], name='moments')
    else:
        batch_mean, batch_var = tf.nn.moments(input, [0], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        updates = tf.group(*ema_apply_op)
        with tf.control_dependencies([updates]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = tf.cond(training,
                        mean_var_with_update,
                        lambda: (batch_mean, batch_var))
    return tf.nn.batch_normalization(input, mean, var, beta, scale, variance_epsilon=1e-3, name=name)