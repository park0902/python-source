# nltk 관련된 punkt가 없을 경우 실행
# import nltk
# nltk.download('punkt')

from nltk.tag import StanfordPOSTagger
from nltk.tokenize import word_tokenize

STANFORD_POS_MODEL_PATH = 'D:\park\Document_Analysis\models\english-bidirectional-distsim.tagger'
STANFORD_POS_JAR_PATH = 'D:\park\Document_Analysis\stanford-postagger-3.8.0.jar'

pos_tagger = StanfordPOSTagger(STANFORD_POS_MODEL_PATH, STANFORD_POS_JAR_PATH)

# 임의로 만든 예제(원하는 문장으로 바꿔서 실습)
text = 'One day in November 2017, the two authors of this book, Seungyeon and Youngjoo, ' \
       'had a coffee at Red Rock cafe, which is very popular place in Mountain View'
tokens = word_tokenize(text)

# 쪼개진 토큰 출력
# print(tokens)
# print()

# 품사 태깅 출력
# print(pos_tagger.tag(tokens))

# 동사와 명사 출력
noun_and_verbs = []
for token in pos_tagger.tag(tokens):
    if token[1].startswith('V') or token[1].startswith('N'):
        noun_and_verbs.append(token[0])
print(', '.join(noun_and_verbs))
