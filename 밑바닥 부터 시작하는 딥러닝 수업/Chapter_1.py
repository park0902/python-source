# 배열 만들기
import numpy as np

a = np.array([[1,2], [3,4]])

print(a)



# a 배열에 모든 원소에 5를 더한 결과
import numpy as np

a = np.array([[1,2], [3,4]])

print(a + 5)



# 배열의 원소들의 평균값 결과
import numpy as np

a = np.array([1,2,4,5,7,10,13,18,21])

print(np.mean(a))



# 배열의 중앙값 결과
import numpy as np

a = np.array([1,2,4,5,7,10,13,18,21])

print(np.median(a))



# 배열의 최대값과 최소값 결과
import numpy as np

a = np.array([1,2,4,5,7,10,13,18,21])

print('최대값 : ',np.max(a), ' 최소값 : ', np.min(a))



# 배열의 표준편차와 분산 결과
import numpy as np

a = np.array([1,2,4,5,7,10,13,18,21])

print('표준편차 : ', np.std(a), ' 분산 : ', np.var(a))



# 행렬식을 numpy 로 구현
import numpy as np

a = np.array([[1,3,7], [1,0,0]])
b = np.array([[0,0,5], [7,5,0]])

print(a+b)




# numpy 배열을 생성하고 원소중에 10만 출력
import numpy as np

a = np.array([[1,2,3], [4,10,6], [8,9,10]])


print(a[1][1])



# 행렬연산을 numpy 로 구현
import numpy as np

a = np.array([[1,2], [3,4]])
b = np.array([10,20])

print(a*b)



# 행렬 연산을 numpy 로 구현
import numpy as np

a = np.array([[0],[10],[20],[30]])
b = np.array([0,1,2])

print(a+b)



# 행렬식 요소에서 15이상인것만 numpy 로 구현
import numpy as np

a = np.array([[51,55],[14,19],[0,4]])

print(a[a>=15])



# 행렬식을 numpy를 이용하지 않고 list 변수로 구현하고 행의 갯수가 몇개인지 출력
a = [[1,3,7], [1,0,0]]

print(len(a))



# 행렬식에서 numpy를 이용하지 않고 list 변수로 구현하고 열의 갯수가 몇개인지 출력
a = [[1,3,7], [1,0,0]]

print(len(a[0]))



# 행렬식의 덧셈 연산을 numpy 이용하지 않고 출력
a = [[1,3,7], [1,0,0]]
b = [[0,0,5], [7,5,0]]
c = [[0,0,0], [0,0,0]]
for i in range(len(a)):
    for j in range(len(a[0])):
        c[i][j] = a[i][j] + b[i][j]

print(c)



# 행렬식의 곱셈 연산을 numpy 이용하지 않고 구현

# 넘파이 O
import numpy as np

a = np.matrix([[1,2], [3,4]])
b = np.matrix([[5,6], [7,8]])

print(a*b)

# 넘파이 X
a = [[1,2], [3,4]]
b = [[5,6], [7,8]]
c = [[0,0], [0,0]]

for i in range(len(a)):
    for j in range(len(b[0])):
        for k in range(len(a[0])):
            c[i][j] += a[i][k] * b[k][j]

print(c)




# 행렬식의 뺄셈 연산을 numpy 와 numpy를 이용하지 않았을때 2가지 방법 구현
import numpy as np

a = np.matrix([[10,20], [30,40]])
b = np.matrix([[5,6], [7,8]])

print(a-b)

a = [[10,20], [30, 40]]
b = [[5,6], [7,8]]
c = [[0,0], [0,0]]

for i in range(len(a)):
    for j in range(len(b[0])):
        c[i][j] += a[i][j] - b[i][j]

print(c)



# 행렬 연산을 numpy 와 numpy를 이용하지 않았을때 2가지 방법 구현
import numpy as np

a = np.array([[1,2], [3,4]])
b = np.array([10,20])

print(a*b)



# numpy의 브로드캐스트를 사용한 연산을 numpy를 이용하지 않는 방법으로 구현
a = [[1,2], [3,4]]
b = [[10,20]]
c = [[0,0], [0,0]]

for i in range(len(a)):
    for j in range(len(b[0])):
        c[i][j] = a[i][j] * b[0][j]

print(c)



# 예제 1
import matplotlib.pyplot as plt

plt.figure()    # 객체 선언
plt.plot([1,2,3,4,5,6,7,8,9,8,7,6,5,4,3,2,1,0])
plt.show()



# 예제 2
import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0,12,0.01)

plt.figure()
plt.plot(t)
plt.grid()
plt.xlabel('size');plt.ylabel('cost')
plt.title('size & cost')
plt.show()



# numpy 배열로 산포도 그래프 그리기
import matplotlib.pyplot as plt
import numpy as np

x = np.array([0,1,2,3,4,5,6,7,8,9])
y = np.array([9,8,7,9,8,3,2,4,3,4])

plt.scatter(x,y)



# 치킨집 년도별 창업건수를 가지고 라인 그래프 그리기
import matplotlib.pyplot as plt
import numpy as np

chi = np.loadtxt("d:\data\창업건수.csv", skiprows=1, unpack=True, delimiter=',')

x = chi[0]
y = chi[4]

plt.grid()
plt.plot(x, y, marker="o", label='CNT')
plt.xlabel('Chicken_CNT');plt.ylabel('Year')
plt.title('Chicken_CNT & Year')
plt.legend(loc=2)



# 치킨집 년도별 폐업건수를 가지고 라인 그래프 겹치게 해서 그리기
import matplotlib.pyplot as plt
import numpy as np

chi1 = np.loadtxt("d:\data\창업건수.csv", skiprows=1, unpack=True, delimiter=',')
chi2 = np.loadtxt("d:\data\폐업건수.csv", skiprows=1, unpack=True, delimiter=',')

x1 = chi1[0];y1 = chi1[4]
x2 = chi2[0];y2 = chi2[4]

plt.figure(figsize=(6,4))
plt.grid()
plt.plot(x1, y1, marker="o", label='OPEN')
plt.plot(x2, y2, marker="o", label='CLOSE')
plt.xlabel('Chicken_Store_CNT');plt.ylabel('Year')
plt.title('Chicken_Store_CNT & Year')
plt.legend(loc=1)




# 이미지 표시하기
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread

img = imread('d:\data\lena.png')

plt.imshow(img)

# plt.show()



# 고양이 사진 표시하기
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread

img = imread('d:\data\\40689.jpg')

plt.imshow(img);plt.hold(True)

plt.show()







