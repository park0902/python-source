import sys
import time
import glob
import unicodedata
from konlpy.tag import Mecab
from gensim.models import Word2Vec

# parameters
WINDOW=5
EMBEDDING_SIZE=200
BATCH_SIZE=10000
ITER=10

def read_text(fin):
    # 전처리된 위키백과 파일 읽기
    corpus_li = []
    mecab = Mecab(dicpath='')
    for line in open(fin):
        # 깨지는 글자를 처리하기 위해 unicodedata.normalize 함수를 이용해
        # NFKC로 변환
        line = unicodedata.normalize('NFKC', line)
        try:
            # 첫 글자가 숫자로 시작하는 문장을 말뭉치에 추가
            _ = int(line[0])
            corpus_li.append(' '.join(mecab.nouns(line)) + '\n')
        except ValueError:
            # 첫 글자가 한글로 시작하는 문장을 말뭉치에 추가
            if ord(line[0]) >= ord('가') and ord(line[0]) <= ord('힇'):
                corpus_li.append(' '.join(mecab.nouns(line)) + '\n')
            else:
                pass

    print('# of lines in corpus', len(corpus_li))
    return (corpus_li)

def train_word2vec(corpus_li, fout_model):
    # read_text에서 생성한 말뭉치를 이용해 word2vec을 학습시킨다
    model = Word2Vec(corpus_li, sg=1, size=EMBEDDING_SIZE, window=WINDOW,
                     min_count=5, workers=3, batch_words=BATCH_SIZE, iter=ITER)
    model.init_sims(replace=True)   # 메모리 정리
    model.save(fout_model)
    return (model)

# 전처리된 파일을 한번에 읽어 들이기 위한 정규식
    input_pattern = 'kowiki-latest-pages-articles.xml-*.txt'
    fin_li = glob.glob(input_pattern)
    for fin in fin_li:
        corpus_li = read_text(fin)
    model = train_word2vec(corpus_li, NAME_FOR_MODEL)




mecab = Mecab(dicpath='D:\park\Document_Analysis\mecab')
mecab.pos('베이라영양')