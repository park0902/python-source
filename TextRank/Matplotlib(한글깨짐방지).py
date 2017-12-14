from konlpy.corpus import kobill
from konlpy.tag import Twitter; t = Twitter()
import nltk

#####################################################################
# matplotlib 에서 한글 깨짐 방지 코드
#####################################################################
from matplotlib import font_manager, rc
font_fname = 'c:/windows/fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname=font_fname).get_name()
rc('font', family=font_name)
#####################################################################

file_ko = kobill.fileids()
doc_ko = kobill.open('1809890.txt').read()
tokens_ko = t.morphs(doc_ko)
ko = nltk.Text(tokens_ko, name='대한민국 국회 의안 제 1809890호')

# print(len(ko.tokens))
# print(len(set(ko.tokens)))
# print(ko.vocab())

ko.plot(50)
ko.concordance('초등학교')