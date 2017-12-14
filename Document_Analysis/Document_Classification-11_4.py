import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

spam_header = 'spam\t'
no_spam_header = 'ham\t'
documents = []
lables = []

with open('D:\\park\\Document_Analysis\\SMSSpamCollection', 'r', encoding='UTF8') as file_handle:
    for line in file_handle:
        # 각 줄에서 레이블 부분만 떼어내고 나머지를 documents에 넣는다
        if line.startswith(spam_header):
            lables.append(1)
            documents.append(line[len(spam_header):])
        elif line.startswith(no_spam_header):
            lables.append(0)
            documents.append(line[len(no_spam_header):])

vectorizer = CountVectorizer()  # 단어 횟수 피처 클래스
term_counts = vectorizer.fit_transform(documents)   # 문서에서 단어 횟수를 센다
vocabulary = vectorizer.get_feature_names()

# 단어 횟수 피처에서 단어 빈도 피처를 만드는 클래스
# tf-idf 에서 idf를 생성하지 않으면 term frequency 생성
tf_transformer = TfidfTransformer(use_idf=False).fit(term_counts)
features = tf_transformer.transform(term_counts)

# 처리된 파일 저장
with open('D:\\park\\Document_Analysis\\processed.pikle', 'wb') as file_handle:
    pickle.dump((vocabulary, features, lables), file_handle)