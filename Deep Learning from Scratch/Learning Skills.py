'''
--------------------------------------------------------------------------------------
- 확률적 경사 하강법(Stochastic Gradient Descent)

                 aL
    W <- W - n ------       W : 갱신할 가중치 매개변수
                 aW         
                           aL
                          ---- : W 에 대한 손실함수의 기울기
                           aW
                           
                            n : 학습률(실제로는 0.01 나 0.001 과 같은 값을 미리 정해서 사용)
                            
                            
    SGD 단점 : 비등방성(anisotropy) 함수(방향에 따라 성질, 즉 여기에서는 기울기가 달라지는 함수)에서는 탐색 경로가 비효율적!!
--------------------------------------------------------------------------------------
'''

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.key():
            params[key] -= self.lr * grads[key]




'''
--------------------------------------------------------------------------------------
- 모멘텀

    운동량을 뜻하는 단어로 물리와 관계가 있다
    
                  aL
    v <- av - n ------        W : 갱신할 가중치 매개변수
                  aW
                            aL
                           ---- : W 에 대한 손실함수의 기울기
    W <- W + v              aW      
                                                               
                             n : 학습률(실제로는 0.01 나 0.001 과 같은 값을 미리 정해서 사용)       
                             
                             v : 속도
                             
                          av항 : 물체가 아무런 힘을 받지 않을 때 서서히 하강시키는 역할 (a는 0.9 등의 값으로 설정)  
--------------------------------------------------------------------------------------
'''

import numpy as np

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)    # 해당 변수의 구조대로 생성

        for key in params.key():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]




'''
--------------------------------------------------------------------------------------
- AdaGrad

    각각의 매개변수에 맞게 맞춤형 값을 만들어준다!
    개별 매개변수에 적응적으로 학습률을 조정하면서 학습을 진행
    

    신경망 학습에서는 학습률(수식에서는 η 으로 표시) 값이 중요!
    
    이 값이 너무 작으면 학습 시간이 너무 길어지고, 반대고 너무 크면 발산하여 학습이 제대로 이뤄지지 않는다
    
    학습률 감소(learning rate decay) : 학습을 진행하면서 학습률을 점차 줄여가는 방법
    
               aL        aL                              1
    h <- h + ------  * ------        매개변수를 갱신할 때 ------ 을 곱해서 학습률 조정!
               aW        aW                            루트 h
               
                  1        aL
    W <- W - η ------- * ------
                루트 h      aW
                
    매개변수의 원소 중에서 많이 움직인(크게 갱신된) 원소는 학습률이 낮아진다는 뜻인데,
    다시 말해서 학습률 감소가 매개변수의 원소마다 다르게 적용됨!
--------------------------------------------------------------------------------------
'''

class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, parames, grads):
        if self.h is None:
            self.h = {}
            for key, val in parames.items():
                self.h[key] = np.zeros_like(val)

        for key in parames.key():
            self.h[key] += grads[key] * grads[key]
            parames[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


