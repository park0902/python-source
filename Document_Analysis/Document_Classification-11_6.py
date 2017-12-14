from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

spam_header = 'spam\t'
no_spam_header = 'ham\t'
documents = []

# 단순 문서 추출
with open('D:\\park\\Document_Analysis\\SMSSpamCollection', 'r', encoding='UTF8') as file_handle:
    for line in file_handle:
        if line.startswith(spam_header):
            documents.append(line[len(spam_header):])
        elif line.startswith(no_spam_header):
            documents.append(line[len(no_spam_header):])

# LDA는 단어 빈도 피처보다 개수 피처가 잘 작동된다
# CountVectorizer를 사용, 토픽 모델에 도움이 되지 않는
# 단어(stop_words)를 자동으로 제거
vectorizer = CountVectorizer(stop_words='english', max_features=2000)
term_counts = vectorizer.fit_transform(documents)
vocabulary = vectorizer.get_feature_names()

# 토픽 모델 학습
topic_model = LatentDirichletAllocation(n_topics=10)
topic_model.fit(term_counts)

# 학습된 토픽을 하나씩 출력
topics = topic_model.components_
for top_id, weights in enumerate(topics):
    print('topic %d' % top_id, end=': ')
    pairs = []
    for term_id, value in enumerate(weights):
        pairs.append((abs(value), vocabulary[term_id]))
    pairs.sort(key=lambda x:x[0], reverse=True)
    for pair in pairs[:10]:
        print(pair[1], end=',')
    print()