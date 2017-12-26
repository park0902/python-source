# # -*- coding: utf-8 -*-
# ''' This program takes a excel sheet as input where each row in first column of sheet represents a document.  '''
#
# import pandas as pd
# import numpy as np
#
# data = pd.read_excel('C:\\Users\medicisoft\Downloads\\data.xlsx',dtype=str)  # Include your data file instead of data.xlsx
#
# idea = data.iloc[:, 0:1]  # Selecting the first column that has text.
#
# # Converting the column of data from excel sheet into a list of documents, where each document corresponds to a group of sentences.
# corpus = []
# for index, row in idea.iterrows():
#     corpus.append(row['Idea'])
#
# '''Or you could just comment out the above code and use this dummy corpus list instaed if don't have the data.
#
# corpus=['She went to the airport to see him off.','I prefer reading to writing.','Los Angeles is in California. It's southeast of San Francisco.','I ate a burger then went to bed.','Compare your answer with Tom's.','I had hardly left home when it began to rain heavily.','If he had asked me, I would have given it to him.
# ','I could have come by auto, but who would pay the fare? ','Whatever it may be, you should not have beaten him.','You should have told me yesterday','I should have joined this course last year.','Where are you going?','There are too many people here.','Everyone always asks me that.','I didn't think you were going to make it.','Be quiet while I am speaking.','I can't figure out why he said so.'] '''
#
# # Count Vectoriser then tidf transformer
#
# from sklearn.feature_extraction.text import CountVectorizer
#
# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(corpus)
#
# # vectorizer.get_feature_names()
#
# # print(X.toarray())
#
# from sklearn.feature_extraction.text import TfidfTransformer
#
# transformer = TfidfTransformer(smooth_idf=False)
# tfidf = transformer.fit_transform(X)
# # print(tfidf.shape)
#
# from sklearn.cluster import KMeans
#
# num_clusters = 5  # Change it according to your data.
# km = KMeans(n_clusters=num_clusters)
# km.fit(tfidf)
# clusters = km.labels_.tolist()
#
# idea = {'Idea': corpus, 'Cluster': clusters}  # Creating dict having doc with the corresponding cluster number.
# frame = pd.DataFrame(idea, index=[clusters], columns=['Idea', 'Cluster'])  # Converting it into a dataframe.
#
# print("\n")
# print(frame)  # Print the doc with the labeled cluster number.
# print("\n")
# print(frame['Cluster'].value_counts())  # Print the counts of doc belonging to ach cluster.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
num_vectors = 1000
num_clusters = 3
num_steps = 100
vector_values = []
for i in range(num_vectors):
  if np.random.random() > 0.5:
    vector_values.append([np.random.normal(0.5, 0.6),
                          np.random.normal(0.3, 0.9)])
  else:
    vector_values.append([np.random.normal(2.5, 0.4),
                         np.random.normal(0.8, 0.5)])
df = pd.DataFrame({"x": [v[0] for v in vector_values],
                   "y": [v[1] for v in vector_values]})
sns.lmplot("x", "y", data=df, fit_reg=False, size=7)
plt.show()
vectors = tf.constant(vector_values)
centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors),
                                 [0,0],[num_clusters,-1]))
expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroids = tf.expand_dims(centroids, 1)

print(expanded_vectors.get_shape())
print(expanded_centroids.get_shape())

distances = tf.reduce_sum(tf.square(tf.sub(expanded_vectors, expanded_centroids)), 2)
assignments = tf.argmin(distances, 0)


means = tf.concat(0, [
  tf.reduce_mean(
      tf.gather(vectors,
                tf.reshape(
                  tf.where(
                    tf.equal(assignments, c)
                  ),[1,-1])
               ),reduction_indices=[1])
  for c in range(num_clusters)])

update_centroids = tf.assign(centroids, means)
init_op = tf.initialize_all_variables()

#with tf.Session('local') as sess:
sess = tf.Session()
sess.run(init_op)

for step in range(num_steps):
   _, centroid_values, assignment_values = sess.run([update_centroids,
                                                    centroids,
                                                    assignments])
print("centroids")
print(centroid_values)


data = {"x": [], "y": [], "cluster": []}
for i in range(len(assignment_values)):
  data["x"].append(vector_values[i][0])
  data["y"].append(vector_values[i][1])
  data["cluster"].append(assignment_values[i])
df = pd.DataFrame(data)
sns.lmplot("x", "y", data=df,
           fit_reg=False, size=7,
           hue="cluster", legend=False)
plt.show()

