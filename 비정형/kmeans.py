import sys
import pandas as pd
import tensorflow as tf
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
# data = np.loadtxt('D:\park\비정형\\datatest.csv', delimiter=" ", dtype=np.int32)
data = np.loadtxt('D:\park\비정형\\xxx1515460251015.csv', delimiter=" ", dtype=np.int32)
x = data[:,0:-1]    # from 1st to (n-1)th column, when data has n columns
y = data[:,[-1]]    # nth column, when data han n columns

# num_vectors = x.size
num_clusters = 9
num_steps = 100
vector_values = data
# print(num_vectors, vector_values)
vectors = vector_values # tf.constant(vector_values)
#centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0,0], [num_clusters,-1]))
centroids = tf.Variable(tf.slice(vectors, [0,0], [num_clusters,-1]))
expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroids = tf.expand_dims(centroids, 1)

distances = tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)
assignments = tf.argmin(distances, 0)

means = tf.concat([
  tf.reduce_mean(
      tf.gather(vectors, 
                tf.reshape(
                  tf.where(
                    tf.equal(assignments, c)
                  ),[1,-1])
               ),reduction_indices=[1])
  for c in range(num_clusters)], 0)

update_centroids = tf.assign(centroids, means)


init_op = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_op)


for step in range(num_steps):
   _, centroid_values, assignment_values = sess.run([update_centroids, centroids, assignments])

print("Centroids")
for i in range(len(centroid_values)):
    print(centroid_values[i])

print()

print("Clustering Result")
data = {"x": [], "y": [], "cluster": []}
for i in range(len(assignment_values)):
    data["x"].append(vector_values[i][0])
    data["y"].append(vector_values[i][1])
    data["cluster"].append(assignment_values[i])

print("XValues");
print(data["x"])
print("YValues");
print(data["y"])
print("CValues");
print(data["cluster"])

# df = pd.DataFrame(data)
# sns.lmplot("x", "y", data=df, fit_reg=False, size=7, hue="cluster", legend=False)
# plt.show()



