# Sklearn 패키지(데이터셋, 전처리, PCA)
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# matplotlib 패키지(그래프)
import matplotlib.pyplot as plt
# import mglearn
#
# #####################################################################
# # matplotlib 에서 한글 깨짐 방지 코드
# #####################################################################
# from matplotlib import font_manager, rc
# font_fname = 'c:/windows/fonts/malgun.ttf'
# font_name = font_manager.FontProperties(fname=font_fname).get_name()
# rc('font', family=font_name)
# #####################################################################
#
# # 유방함 데이터셋 로드
# cancer = load_breast_cancer()
#
# # 각 특성의 분산이 1이 되도록 데이터 스케일 작업
# scalar = StandardScaler()
# scalar.fit(cancer.data)
# X_scaled = scalar.transform(cancer.data)
#
# # 데이터의 처음 두 개의 주성분만 유지
# pca = PCA(n_components=2)
#
# # 유방암 데이터로 PCA 모델 생성
# pca.fit(X_scaled)
#
# # 처음 두 개의 주성분을 사용해 데이터 변환
# X_pca = pca.transform(X_scaled)
#
# print("원본 데이터 형태 : {}".format(str(X_scaled.shape)))
# print("축소된 데이터 형태 : {}".format(str(X_pca.shape)))
#
# # 클래스를 색깔로 구분하여 처음 두 개의 주성분을 그래프로 표현
# plt.figure(figsize=(8,8))
# mglearn.discrete_scatter(X_pca[:,0], X_pca[:,1], cancer.target)
# plt.legend(["악성", "양성"], loc="best")
# plt.gca().set_aspect("equal")
# plt.xlabel("첫 번째 주성분")
# plt.ylabel("두 번째 주성분")
# plt.show()
#
# print("PCA 주성분 형태 : {}".format(pca.components_.shape))
#
# print("PCA 주성분 : \n{}".format(pca.components_))
#
#
# # 유방암 데이터셋에서 찾은 처음 두 개의 주성분 히트맵
# plt.matshow(pca.components_, cmap="viridis")
# plt.yticks([0,1], ["첫 번째 주성분", "두 번째 주성분"])
# plt.colorbar()
# plt.xticks(range(len(cancer.feature_names)),
#            cancer.feature_names, rotation=60, ha="left")
# plt.xlabel("특성")
# plt.ylabel("주성분")
# plt.show()


# import pandas as pd
#
# raw_data = {'col0': ['A', 'B', 'C', 'D'],
#             'col1': [10, 20, 30, 40],
#             'col2': [100, 200, 300, 400]}
#
# data = pd.DataFrame(raw_data)
# print(data)
#
# onehot_data = pd.get_dummies(data)
# print(onehot_data)
#
#
# data1 = pd.DataFrame({'col0': ['A', 'B', 'C', 'D'],
#                       'col1': [10, 20, 30, 40],
#                       'col2': [100, 200, 300, 400]})
#
# data1['col1'] = data1['col1'].astype(str)
# data1['col2'] = data1['col2'].astype(str)
#
# onehot_data1 = pd.get_dummies(data1, columns=['col0', 'col1', 'col2'])
# print(onehot_data1)

# from sklearn.preprocessing import OneHotEncoder
# import numpy as np
#
# data = OneHotEncoder()
#
# X = np.array([[0], [1], [2]])
#
# print(X)
#
# data.fit(X)
#
# # 최대 클래스 수
# print(data.n_values_)
#
# # 입력이 벡터인 경우 각 원소를 나타내는 slice 정보
# print(data.feature_indices_)
#
# # 실제로 사용된 클래스
# print(data.active_features_)
#
# # one-hot encoding
# print(data.transform(X).toarray())




# import pandas as pd
# import numpy as np
#
# # 결측값 데이터 생성
# df = pd.DataFrame(np.random.randn(5, 3), columns = ['C1', 'C2', 'C3'])
# df.ix[0, 0] = None
# df.ix[1, ['C1', 'C3']] = np.nan
# df.ix[2, 'C2'] = np.nan
# df.ix[3, 'C1'] = np.nan
# df.ix[3, 'C2'] = np.nan
# df.ix[3, 'C3'] = np.nan
#
# print(df)
#
# # NaN이 하나라도 들어간 행 전체 제거
# print(df.dropna())
# # 모든 데이터가 NaN인 행 전체 제거
# print(df.dropna(how='all'))
#
# # NaN 값을 특정 숫자로 대채
# print(df.fillna(0))
#

# NaN 값을 평균으로 대체
# print(df.fillna(df.mean()))


# import pandas as pd
#
# boston = pd.read_csv("D:\park\data\\boston.csv")
#
# print(boston.head())
#
# corr = boston.corr(method='pearson')
# print(corr)


# import numpy as np
# import pandas as pd
#
# boston = pd.read_csv("D:\park\data\\boston.csv")
# boston_TAX = pd.DataFrame(boston, columns=['TAX'])
# boston_RAD = pd.DataFrame(boston, columns=['RAD'])
#
# TAX = np.array(boston_TAX).reshape(1,506)
# RAD = np.array(boston_RAD).reshape(1,506)
#
# corr = np.corrcoef(TAX,RAD)
#
# print(corr)


# from scipy.stats import chisquare
# import numpy as np
#
# # 데이터 개수
# N = 100
#
# # 모수 개수
# K = 4
#
# # 모수
# theta_0 = np.array([0.35, 0.30, 0.20, 0.15])
# np.random.seed(0)
#
# x = np.random.choice(K, N, p=theta_0)
# n = np.bincount(x, minlength=K)
#
# chi = chisquare(n)
#
# print(chi)


import numpy as np

# (2-5) 카이제곱분포 (Chisq-distribution)로 부터 난수 생성
# Draw samples from a chi-square distribution
# np.random.chisquare(df, size=None)
# df : Number of degrees of freedom

# 난수 생성 초기값 부여
np.random.seed(100)

rand_chisq = np.random.chisquare(df=2, size=20)

print(rand_chisq)






# F-분포 (F-distribution)으로부터 난수 생성
# Draw samples from an F-distribution (Fisher distribution)
# numpy.random.f(dfnum, dfden, size=None)
# dfnum : degrees of freedom in numerator
# dfden : degrees of freedom in denominator

# 난수 생성 초기값 부여
# np.random.seed(100)
#
# rand_f = np.random.f(dfnum=5, dfden=10, size=20)
#
# print(rand_f)





# np.random.randint : Discrete uniform distribution, yielding integers
# np.random.randint(low=0.0, high=1.0, size=None)
# low : Lower boundary of the output interval
# high : Upper boundary of the output interval
# [low, high) : includes low, excludes high

# 난수 생성 초기값 부여
# np.random.seed(100)
#
# rand_int = np.random.randint(low=0, high=10 + 1, size=20)
#
# print(rand_int)






# 균등분포 (Uniform Distribution)로 부터 난수 생성
# Draw samples from a uniform distribution
# np.random.uniform(low=0.0, high=1.0, size=None)
# low : Lower boundary of the output interval
# high : Upper boundary of the output interval
# [low, high) : includes low, excludes high

# 난수 생성 초기값 부여
# np.random.seed(100)
#
# rand_unif = np.random.uniform(low=0.0, high=10.0, size=20)
#
# print(rand_unif)



# t-분포 (Student's t-distribution)로부터 난수 생성
# Draw samples from a standard Student’s t distribution with df degrees of freedom
# np.random.standard_t(df, size=None)
# df : Degrees of freedom
# size : Output shape

# 난수 생성 초기값 부여
# np.random.seed(100)
#
# rand_t = np.random.standard_t(df=3, size=20)
#
# print(rand_t)



# 연속형 확률분포 (continuous probability distribution)
# 정규분포(normal distribution)로부터 난수 생성
# Draw random samples from a normal (Gaussian) distribution
# np.random.normal(loc=0.0, scale=1.0, size=None)
# mu : Mean (“centre”) of the distribution
# sigma : Standard deviation (spread or “width”) of the distribution
# size : Output shape

# 난수 생성 초기값 부여
# np.random.seed(100)
#
# # 평균, 표준편차
# mu, sigma = 0.0, 3.0
#
# rand_norm = np.random.normal(mu, sigma, size=20)
#
# print(rand_norm)



# 포아송 분포 (Poisson Distribution)
# np.random.poisson(lam=1.0, size=None)
# Poisson distribution is the limit of the binomial distribution for large N

# 난수 생성 초기값 부여
# np.random.seed(seed=100)
#
# rand_pois = np.random.poisson(lam=20, size=100)
#
# print(rand_pois)



# 초기하분포 (Hypergeometric distribution)
# 비복원 추출(sampling without replacement)
# np.random.hypergeometric(ngood, nbad, nsample, size=None)

# 난수 생성 초기값 부여
# np.random.seed(seed=100)
#
# rand_hyp = np.random.hypergeometric(ngood=5, nbad=20, nsample=5, size=50)
#
# print(rand_hyp)



# 이항분포 (Binomial Distribution)
# np.random.binomial(n, p, size)
# 복원 추출 (sampling with replacement)
# n an integer >= 0 and p is in the interval [0,1]

# binomial = np.random.binomial(n=1, p=0.5, size=20)
#
# print(binomial)



