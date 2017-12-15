from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

STANFORD_NER_CLASSIFER_PATH = 'D:\park\Document_Analysis\\ner\classifiers\english.muc.7class.distsim.crf.ser.gz'
STANFORD_NER_JAR_PATH = 'D:\park\Document_Analysis\\ner\stanford-ner-3.8.0.jar'

ner_tagger = StanfordNERTagger(STANFORD_NER_CLASSIFER_PATH, STANFORD_NER_JAR_PATH)

# 임의로 만든 예제(원하는 문장으로 바꿔서 실습)
text = 'One day in November 2017, the two authors of this book, Seungyeon and Youngjoo, ' \
       'had a coffee at Red Rock cafe, which is very popular place in Mountain View'

tokens = word_tokenize(text)
# print(ner_tagger.tag(tokens))

# 장소 정보 단어 출력
all_locations = []
for token in ner_tagger.tag(tokens):
    if token[1] == "LOCATION":
        all_locations.append(token[0])

print(', '.join(all_locations))
