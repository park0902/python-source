import pickle
from sklearn.linear_model import LogisticRegression
import numpy as np

# 단어집을 처리합니다.
vocabulary = {}  # 딕셔너리를 선언합니다.
with open('D:\\park\\Document_Analysis\\SMSSpamCollection', 'r', encoding='UTF8') as file_handle: # 파일을 엽니다.
    for line in file_handle:   # 파일을 한 줄씩 읽습니다
        splits = line.split()  # 한 줄을 빈 칸으로 쪼개서 리스트로 만듭니다.
        label = splits[0]      # 맨 앞의 단어는 레이블이니까 따로 둡니다.
        text = splits[1:]

        # 전체 내용을 단어 단위로 살펴보고
        # 사전에 해당 단어가 없으면 추가 후 고유번호를 붙입니다.
        # 그리고 그 매핑을 vocabulary에 저장합니다({단어 -> 고유ID}).
        for word in text:
            lower = word.lower()
            if not lower in vocabulary:
                vocabulary[lower] = len(vocabulary)


# 각 문서의 피처 벡터를 뽑아서 features 리스트에 넣습니다.
features = []
with open('D:\\park\\Document_Analysis\\SMSSpamCollection', 'r', encoding='UTF8') as file_handle:
    for line in file_handle:                 # 파일을 한 줄씩 읽습니다.
        splits = line.split()
        feature = np.zeros(len(vocabulary))  # 0으로 채워진 numpy 벡터를 만듭니다
        text = splits[1:]
        for word in text:
            lower = word.lower()
            # vocabulary에 따라 각 피처가 몇 번 나왔는지 개수를 셉니다
            feature[vocabulary[lower]] += 1

        # 단어 빈도 피처이므로 문서에서 나온 총 단어 수로 전체 벡터를 나누어 피처를 만듭니다.
        feature = feature / sum(feature)
        features.append(feature)


# 레이블을 처리합니다.
labels = []
with open('D:\\park\\Document_Analysis\\SMSSpamCollection', 'r', encoding='UTF8') as file_handle:
    for line in file_handle:  # 파일을 한 줄씩 읽습니다
        splits = line.split()
        label = splits[0]
        if label == 'spam':  # 맨 앞 단어(label)가 spam이면 1, 아니면 0을 추가합니다.
            labels.append(1)
        else:
            labels.append(0)

with open('D:\\park\\Document_Analysis\\processed.pikle', 'rb') as file_handle:
    vocabulary, features, labels = pickle.load(file_handle)


# 학습-평가 데이터 나누기
# 처음 50% 학습, 나머지 평가
total_number = len(labels)
middle_index = total_number//2
train_features = features[:middle_index,:]
train_labels = labels[:middle_index]
test_features = features[middle_index:,:]
test_labels = labels[middle_index:]

classifier = LogisticRegression()
classifier.fit(train_features, train_labels)
# print('train accuracy: %4.4f' % classifier.score(train_features, train_labels))
# print('test accuracy: %4.4f' % classifier.score(test_features, test_labels))



# 어떤 항목이 판별에 영향을 많이 줬는지 찾아보기
weights = classifier.coef_[0, :]
pairs = []
for index, value in enumerate(weights):
    pairs.append( (abs(value), vocabulary[index]))

pairs.sort(key=lambda x:x[0], reverse=True)
print(pairs)
for pair in pairs[:20]:
    print('score %4.4f word: %s' % pair)
