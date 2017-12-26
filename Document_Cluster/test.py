# 샘플데이터 생성
import numpy as np

num_points = 2000
vectors_set = []

for i in range(num_points):
    if np.random.random() > 0.5:
        vectors_set.append([np.random.normal(0.0, 0.9),
                            np.random.normal(0.0, 0.9)])

    else:
        vectors_set.append([np.random.normal(3.0, 0.5),
                            np.random.normal(1.0, 0.5)])


        # matplotlib 기반으로하는 seaborn 시각화 패키지 , 데이터 조작 패키지 pandas
        # seaborn 은 Anaconda 에서 conda install seaborn 명령으로 설치
# pandas 는 pip install pandas 명령으로 설치

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.DataFrame({"x": [v[0] for v in vectors_set],
                   "y": [v[1] for v in vectors_set]})
sns.lmplot("x", "y", data=df, fit_reg=False, size=6)
plt.show()

# K-means 구현
# 4개의 군집으로 그룹화

import tensorflow as tf

# 모든 데이터를 상수 텐서로 옮김
vectors = tf.constant(vectors_set)
# 초기 단계 : 중심 k(4)개를 입력데이터에서 무작위로 선택
k = 4
centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0, 0], [k, -1]))
# vector.get_shape(), centroides.get_shape()
# 위 주석으로 각 텐서의 구조를 확인해볼 수 있음

expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroides = tf.expand_dims(centroides, 1)

# 할당 단계 : 유클리드 제곱거리 사용
diff = tf.sub(expanded_vectors, expanded_centroides)
sqr = tf.square(diff)
distances = tf.reduce_sum(sqr, 2)
assignments = tf.argmin(distances, 0)

# 업데이트 : 새로운 중심 계산
means = tf.concat(0,
                  [tf.reduce_mean(
                      tf.gather(vectors,
                                tf.reshape(
                                    tf.where(tf.equal(assignments, c))
                                    , [1, -1])
                                )
                      , reduction_indices=[1]) for c in range(k)])

update_centroides = tf.assign(centroides, means)

init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)

for step in range(100):
    _, centroid_values, assignment_values = sess.run([update_centroides, centroides, assignments])


    # assignment_values 텐서의 결과를 확인

data = {"x": [], "y": [], "cluster": []}

for i in range(len(assignment_values)):
    data["x"].append(vectors_set[i][0])
    data["y"].append(vectors_set[i][1])
    data["cluster"].append(assignment_values[i])

df = pd.DataFrame(data)
sns.lmplot("x", "y", data=df, fit_reg=False, size=6, hue="cluster", legend=False)
plt.show()


means = tf.concat(0,
                  [tf.reduce_mean(
                      tf.gather(vectors,
                                tf.reshape(
                                    tf.where(tf.equal(assignments, c))
                                    , [1, -1])
                                )
                      , reduction_indices=[1]) for c in range(k)])