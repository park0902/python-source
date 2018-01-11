# TextRank를 이용한 문서요약

'''
* 패키지 설명

Newspaper : 사용자가 입력한 url에서 text를 크롤링 해주는 패키지

KoNLPy : 한글 형태소 분석기로 TextRank를 적용하기 위한 전처리 과정으로 사용되는 패키지
         KoNLPy를 설치하기 위해서는 JPype1 설치

Scikit-learn : 대표적인 머신러닝 패키지 중 하나로 TF-IDF 모델을 생성하는데 사용
'''

from newspaper import Article
from konlpy.tag import Kkma
from konlpy.tag import Twitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import numpy as np


# [텍스트 크롤링, 문장 단위 분리, 명사추출] 과정
class SentenceTokenizer(object):
    def __init__(self):
        self.kkma = Kkma()
        self.twitter = Twitter()
        self.stopwords = ['중인', '만큼', '마찬가지', '꼬집었', '연합뉴스', '데일리', '동아일보', '중앙일보', '조선일보', '기자',
                          '아', '휴', '아이구', '아이쿠', '아이고', '어', '나', '우리', '저희', '따라', '의해', '을',
                          '를', '에', '의', '가', ]


    '''
    url 주소를 받아 기사내용(article.text)을 추출하여 Kkma.sentences()를 이용하여
    문장단위로 나누어 준 후 sentences를 return 해 준다
    '''
    def url2sentence(self, url):
        article = Article(url, languge='ko')
        article.download()
        article.parse()
        sentences = self.kkma.sentences(article.text)

        for idx in range(0, len(sentences)):
            if len(sentences[idx]) <= 10:
                sentences[idx-1] += (' ' + sentences[idx])
                sentences[idx] = ''

        return sentences

    '''
    text(str)를 입력받아 Kkma.sentences()를 이용하여 문장단위로 나누어 준 후
    sentences를 return 해 준다
    '''
    def text2setences(self, text):
        sentences = self.kkma.sentences(text)
        for idx in range(0, len(sentences)):
            if len(sentences[idx]) <= 10:
                sentences[idx-1] += (' ' + sentences[idx])
                sentences[idx] = ''

        return sentences

    '''
    sentences를 받아 Twitter.nouns()를 이용하여 명사를 추출한 뒤 nouns를 return 해 준다
    '''
    def get_nous(self, sentences):
        nouns = []
        for sentence in sentences:
            if sentence is not '':
                nouns.append(' '.join([noun for noun in self.twitter.nouns(str(sentence))
                                       if noun not in self.stopwords and len(noun) > 1]))

        return nouns


# [TF-IDF 모델, 그래프 생성] 과정
class GrapMatrix(object):
    def __init__(self):
        self.tfidf = TfidfVectorizer()
        self.cnt_vec = CountVectorizer()
        self.graph_sentences = []

    '''
    명사로 이루어진 문장을 입력받아 sklearn 의 TfidfVectorizer.fit_transform 을 이용하여
    tfidf matrix를 만든 후 Sentence graph 를 return 한다
    '''
    def bulid_sent_graph(self, sentence):
        tfidf_mat = self.tfidf.fit_transform(sentence).toarray()
        self.graph_sentences = np.dot(tfidf_mat, tfidf_mat.T)

        return self.graph_sentences


    '''
    명사로 이루어진 문장을 입력받아 sklearn의 CountVectorizer.fit_transform을 이용하여 matrix를 만든 후
    word graph 와 {idx: word}형태의 dictionany를 return 한다
    '''
    def bulid_words_graph(self, sentence):
        cnt_vec_mat = normalize(self.cnt_vec.fit_transform(sentence).toarray().astype(float), axis=0)
        vocab = self.cnt_vec.vocabulary_

        return np.dot(cnt_vec_mat.T, cnt_vec_mat), {vocab[word] : word for word in vocab}


# [TextRank 알고리즘] 구현
class Rank(object):
    def get_ranks(self, graph, d=0.85):     # d = damping factor
        A = graph
        matrix_size = A.shape[0]
        for id in range(matrix_size):
            A[id, id] = 0                   # diagonal 부분을 0으로
            link_sum = np.sum(A[:,id])      # A[:, id] = A[:][id]
            if link_sum != 0:
                A[:, id] /= link_sum
            A[:, id] += -d
            A[id, id] = 1

        B = (1-d) * np.ones((matrix_size, 1))
        ranks = np.linalg.solve(A, B)       # 연립방정식 Ax = b

        return {idx: r[0] for idx, r in enumerate(ranks)}


# [문서의 요약 또는 키워드 확인] 과정
class TextRank(object):
    def __init__(self, text):
        self.sent_tokenize = SentenceTokenizer()

        if text[:5] in ('http:', 'https'):
            self.sentences = self.sent_tokenize.url2sentence(text)
        else:
            self.sentences = self.sent_tokenize.text2setences(text)

        self.nouns = self.sent_tokenize.get_nous(self.sentences)

        self.graph_matrix = GrapMatrix()
        self.sent_graph = self.graph_matrix.bulid_sent_graph(self.nouns)
        self.words_graph, self.idx2word = self.graph_matrix.bulid_words_graph(self.nouns)

        self.rank = Rank()
        self.sent_rank_idx = self.rank.get_ranks(self.sent_graph)
        self.sorted_sent_rank_idx = sorted(self.sent_rank_idx, key=lambda k: self.sent_rank_idx[k], reverse=True)

        self.word_rank_idx = self.rank.get_ranks(self.words_graph)
        self.sorted_word_rank_idx = sorted(self.word_rank_idx, key=lambda k: self.word_rank_idx[k], reverse=True)

    '''
    Default로 3줄 요약 가능하게 구현
    '''
    def summarize(self, sent_num):
        summary = []
        index = []
        for idx in self.sorted_sent_rank_idx[:sent_num]:
                index.append(idx)

        index.sort()
        for idx in index:
            summary.append(self.sentences[idx])

        return summary

    '''
    Default로 10개의 키워드를 출력하도록 구현
    '''
    def keywords(self, word_num = 10):
        rank = Rank()
        rank_idx = rank.get_ranks(self.words_graph)
        sorted_rank_idx = sorted(rank_idx, key=lambda k: rank_idx[k], reverse=True)

        keywords = []
        index = []
        for idx in sorted_rank_idx[:word_num]:
            index.append(idx)

        for idx in index:
            keywords.append(self.idx2word[idx])

        return keywords

print('Wait.....')
print()

############################################################################################
# URL
# url = 'http://news.naver.com/main/read.nhn?mode=LSD&mid=shm&sid1=105&oid=014&aid=0003899247'
# textrank = TextRank(url)
#
# print("Summary")
# for row in textrank.summarize(5):
#     print()
#     print(row)
#
# print()
# print('keywords : ', textrank.keywords())

#############################################################################################
# TXT
f = open('D:\park\TextRank\\news.txt', 'r')
lines = f.read()


textrank = TextRank(lines)

print("Summary")
for row in textrank.summarize(3):
    print()
    print(row)

print()
print('keywords : ', textrank.keywords())

##############################################################################################
# DOCX
# import docx2txt as docx
#
# docx_ = docx.process('D:\park\TextRank\\TextRank 조사 _v0.1.docx')
#
# textrank = TextRank(docx_)
#
# print("Summary")
# for row in textrank.summarize(5):
#     print()
#     print(row)
#
# print()
# print('keywords : ', textrank.keywords())