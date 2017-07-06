'''
--------------------------------------------------------------------------------------
- 4차원 배열

    CNN 에서 계층 사이를 흐르는 데이터는 4차원
    
    예 : 형상 (10,1,28,28) 
    
        높이 : 28     너비 : 28     채널 : 1      데이터 : 10개
--------------------------------------------------------------------------------------
'''

import numpy as np

x = np.random.rand(10,1,28,28)

print(x.shape)          # X 형상
print(x[0].shape)       # X 의 첫번째 데이터 형상
print(x[1].shape)       # X 의 두번째 데이터 형상
print(x[0][0].shape)    # X 의 첫번째 데이터의 첫 채널의 공간




'''
--------------------------------------------------------------------------------------
- im2col 함수

    im2col
    
        입력 데이터를 필터링(가중치 계산)하기 좋게 전개하는(펼치는) 함수
        
        3차원 입력 데이터에 im2col을 적용하면 2차원 행렬로 바뀐다!
        (정확히는 배치 안의 데이터 수까지 포함한 4차원 데이터를 2차원으로 변환!)
        
        im2col로 전개한 후의 원소 수가 원래 블록의 원소 수보다 많아진다
        그래서 im2col을 사용해 구현하면 메모리를 더 많이 소비하는 단점!
        하지만 컴퓨터는 큰 행렬을 묶어서 계산하는 데 탁월
        그래서 문제를 행렬 계산으로 만들면 선형 대수 라이브러리를 활용해 효율을 높일 수 있다!
        
        im2col로 입력 데이터를 전개한 다음에는 합성곱 계층의 필터(가중치)를 1열로 전개하고, 두 행렬의 내적을 계산!
        이는 완전연결 계층의 Affine 계층에서 한 것과 거의 같다!
        
        im2col 방식으로 출력한 결과는 2차원 행렬!
        CNN은 데이터를 4차원 배열로 저장하므로 2차원인 출력 데이터를 4차원으로 변형(reshape)!!
        
        
--------------------------------------------------------------------------------------
'''

import sys, os
import numpy as np
sys.path.append(os.pardir)
from common.util import im2col

x1 = np.random.rand(1, 3, 7, 7)     # (데이터 수 , 채널 수, 높이, 너비)
col1 = im2col(x1, 5, 5, stride=1, pad=0)
print(col1.shape)   # (90, 75)

x2 = np.random.rand(10, 3, 7, 7)    # (데이터 10개)
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape)   # (90, 75)