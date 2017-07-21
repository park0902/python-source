# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image       # 파이썬 이미지 라이브러리


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))    # 파이썬 이미지 객체로 변환
    pil_img.show()

# (훈련데이터, 훈련데이터 라벨), (테스트데이터, 테스트데이터 라벨)
# flatten = True : 입력 이미지를 평탄하게 1차원 배열로 변환
# normalize = True : 입력 이미지의 픽셀의 값을 0~1사이로 정규화(False : 원래값인 0~255 값)
# 전처리 : 신경망의 입력 데이터에 특정 변환을 가하는 것
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

for i in range(100):
    img = x_train[i]
    label = t_train[i]
    print(label, i)


img = x_train[4]
label = t_train[0]



# print(label)  # 5

# print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 형상을 원래 이미지의 크기로 변형
# print(img.shape)  # (28, 28)

# print(len(x_train))
# print(len(x_test))

img_show(img)
