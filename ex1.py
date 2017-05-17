# # from tkinter import *
# #
# # tk = Tk()
# # canvas = Canvas(tk, width=500, height=450)
# # canvas.pack()
# #
# #
# # canvas.create_polygon(0,350,15,350,15,375,30,375,30,350,45,350,45,450,0,450, outline='yellow', fill='green', width=1)
# # canvas.mainloop()
#
#
#
# from tkinter import *
# import random
# import time
#
#
#
# class Man:
#     def __init__(self, canvas):
#         self.canvas = canvas
#         self.man = canvas.create_rectangle(0, 0, 10, 20, fill='black')
#         self.canvas.move(self.man, 295, 480)
#         self.x = 0
#         self.y = 0
#         self.canvas_width = self.canvas.winfo_width()      #man의 이동범위를 canvas의 밑변으로 제한
#         self.canvas_height = self.canvas.winfo_height()    #man의 이동범위를 canvas의 높이로 제한
#         self.canvas.bind_all('<KeyPress-Left>', self.turn_left)
#         self.canvas.bind_all('<KeyPress-Right>', self.turn_right)   #canvas가 감지
#         print(self.canvas_width)
#
#     def draw(self):
#         man_pos = self.canvas.coords(self.man)       #self.man의 좌상우하의 좌표, 위치
#
#         if man_pos[0] <= 0 and self.x < 0:                      #self.man이 오른쪽으로 나가지 않도록
#             self.x = 5
#         elif man_pos[2] >= self.canvas_width and self.x > 0:
#             self.x = -5
#
#         self.canvas.move(self.man, self.x, self.y)
#
#
#     def turn_left(self, evt):
#         self.x = -5
#
#
#     def turn_right(self, evt):
#         self.x = 5
#
# #
# # class poo:
# #     def __init__(self):
# #
# # class candy:
# #     def __init__(self):
#
#
# tk = Tk()
# tk.title("Dodge Your Poop Faster")   #게임 창의 제목 출력
# tk.resizable(0, 0)                   #tk.resizable(가로크기조절, 세로크기조절)
# tk.wm_attributes("-topmost", 1)      #생성된 게임창을 다른창의 제일 위에 오도록 정렬
# tk.update()  # 여기서 한번 다시 적어준다.
#
# canvas = Canvas(tk, width=600, height=500, bd=0, highlightthickness=0)
# #bd=0, highlightthickness=0 은 베젤의 크기를 의미한다.
# canvas.configure(background='#E8D487')
# canvas.pack()  #앞의 코드에서 전달된 폭과 높이는 매개변수에 따라 크기를 맞추라고 캔버스에에 말해준다.
#
# man = Man(canvas)
#
# while 1:
#     tk.update()
#     tk.update_idletasks()
#     man.draw()
#     time.sleep(0.015)

#
# def sum():
#     a = 1
#     b = 2
#     for i in range(5):
#         sum = i + a
#         print(sum)
#
#     return sum
#
# sum()


# import urllib.request                      # 웹 브라우져 html 문서를 얻어오기위해 통신하기 위한 모듈
# from  bs4 import BeautifulSoup
#
# params = []
# def fetch_list_url():
#     for i in range(0,20):
#         list_url = 'http://search.hani.co.kr/Search?command=query&keyword=%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5&media=news&sort=d&period=all&datefrom=2000.01.01&dateto=2017.05.17&pageseq=0'+str(i)
#         url = urllib.request.Request(list_url) # url 요청에 따른 http 통신 헤더값을 얻어낸다
#         res = urllib.request.urlopen(url).read().decode("utf-8") # 영어가 아닌 한글을 담아내기 위한 문자셋인 유니코드 문자셋을
#                                                             # 사용해서 html 문서와 html  문서내의 한글을 res 변수에 담는다
#         soup = BeautifulSoup(res, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
#         # print(i+1, '번째 페이지 입니다.')
#         for j in range(20):
#             try:
#                 a = soup.find_all('dt')[j]
#                 b= a.find("a")["href"]
#                 params.append(b)
#             except Exception:
#                 pass
#         # print(params)
#         # return params
# fetch_list_url()
#
# for i in range(len(params)):
#     print(params[i])




# import urllib.request                      # 웹 브라우져 html 문서를 얻어오기위해 통신하기 위한 모듈
# from  bs4 import BeautifulSoup
#
# def fetch_list_url():
#     for i in range(0,20):
#         list_url = 'http://search.hani.co.kr/Search?command=query&keyword=%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5&media=news&sort=d&period=all&datefrom=2000.01.01&dateto=2017.05.17&pageseq=0'+str(i)
#         url = urllib.request.Request(list_url) # url 요청에 따른 http 통신 헤더값을 얻어낸다
#         res = urllib.request.urlopen(url).read().decode("utf-8") # 영어가 아닌 한글을 담아내기 위한 문자셋인 유니코드 문자셋을
#                                                             # 사용해서 html 문서와 html  문서내의 한글을 res 변수에 담는다
#         soup = BeautifulSoup(res, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
#
#         for j in range(20):
#             try:
#                 a = soup.find_all('dt')[j]
#                 b= a.find("a")["href"]
#                 print(b)
#
#             except Exception:
#                 pass
# fetch_list_url()


# import urllib.request
# from  bs4 import BeautifulSoup
#
# params = []
#
# def fetch_list_url(cnt):
#     list_url = "http://search.hani.co.kr/Search?command=query&keyword=%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5&media=news&sort=d&period=all&datefrom=2000.01.01&dateto=2017.05.17&pageseq={}".format(cnt)
#                 ###여기에 스크롤링할 웹페이지의 url 을 붙여넣습니다.
#
#     url = urllib.request.Request(list_url)                          # url 요청에 따른 http 통신 헤더값을 얻어낸다
#     res = urllib.request.urlopen(url).read().decode("utf-8")        # 영어가 아닌 한글을 담아내기 위한 문자셋인 유니코드 문자셋을
#                                                                     # 사용해서 html 문서와 html  문서내의 한글을 res 변수에 담는다. (유니코드 안쓰면 글씨 다 깨짐)
#     '''
#     참고>>  문자를 담는 set:
#             1. 아스키코드 : 영문
#             2. 유니코드 : 한글, 중국어
#     '''
#     soup = BeautifulSoup(res, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
#     # print(soup)
#     return soup
#
#     soup2 = soup.find_all('p')[0]
#     # print(soup2)
#     return soup2
#
#     soup3 = soup2.find("a")     #print(soup2.a)와 같음
#     # print(soup3)
#     return soup3
#
#     soup3 = soup2.find("a")["href"]     #이렇게 하면 href의 링크만 나옴
#     # print(soup3)
#     return soup3
#
#     for i in range(8):
#         soup2 = soup.find_all('p')[i]
#         # print(soup2.a["href"])
#     soup3 = soup2.find_all('a')     #이렇게 하면 href의 링크만 나옴
#     # print(soup3)
#     return soup3
#
#     for link in soup.find_all('p'):
#         print(link.a["href"])   #print(link.find('a')[href])도 같음
#
#     for link in soup.find_all('dt'):   #p에는 이미지가 있는 뉴스만 있다. 이미지 없는 뉴스도 가져오기 위해서 dt사용
#         try:
#             print(link.a["href"])   #print(link.find('a')[href])도 같음
#         except:                        #<dt></dt>처럼 비어있는 곳이 있기 때문에 try except를 사용해야 한다.
#             continue
#
#         return link.a["href"]
#
#
# for i in range(115):
#     fetch_list_url(i)
#     params.append(fetch_list_url(i))
#
# print(params)




# import urllib.request           # 웹 브라우져에서 html 문서를 얻어오기위해 통신하기 위한 모듈
# from bs4 import BeautifulSoup  # html 문서검색 모듈
#
# def fetch_list_url():
#     for i in range(0,20):
#         list_url = 'http://search.hani.co.kr/Search?command=query&keyword=%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5&media=news&sort=d&period=all&datefrom=2000.01.01&dateto=2017.05.17&pageseq=0'+str(i)
#         url = urllib.request.Request(list_url) # url 요청에 따른 http 통신 헤더값을 얻어낸다
#         res = urllib.request.urlopen(url).read().decode("utf-8")
#         params = []
#         # 영어가 아닌 한글을 담아내기 위한 문자셋인 유니코드 문자셋을 사용해서 html 문서와 html  문서내의 한글을 res 변수에 담는다.
#         # 문자를 받는 set(p256)
#         # 1. US7ASCII : 영문
#         # 2. 유니코드(UTF8) : 한글, 중국어
#
#         soup = BeautifulSoup(res, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
#         soup_p = soup.find_all('dt')
#         p = len(soup_p)
#
#         for j in range(p):
#             try:
#                 soup_p = soup.find_all('dt')[j]
#                 soup_p_a = soup_p.find('a')['href']
#                 params.append(soup_p_a)
#
#             except Exception:
#                 pass
#         print(params)
#         return params
#
# fetch_list_url()







# import sys
# from konlpy.tag import Twitter
# from collections import Counter
#
#
# def get_tags(text, ntags=50):
#     spliter = Twitter()
#     nouns = spliter.nouns(text)
#     count = Counter(nouns)
#     return_list = []
#     for n, c in count.most_common(ntags):
#         temp = {'tag': n, 'count': c}
#         return_list.append(temp)
#     return return_list
#
#
# def main(argv):
#     if len(argv) != 4:
#         print('python [모듈 이름] [텍스트 파일명.txt] [단어 개수] [결과파일명.txt]')
#         return
#     text_file_name = argv[1]
#     noun_count = int(argv[2])
#     output_file_name = argv[3]
#     open_text_file = open(text_file_name, 'r')
#     text = open_text_file.read()
#     tags = get_tags(text, noun_count)
#     open_text_file.close()
#     open_output_file = open(output_file_name, 'w')
#     for tag in tags:
#         noun = tag['tag']
#         count = tag['count']
#         open_output_file.write('{} {}\n'.format(noun, count))
#     open_output_file.close()
#
# if __name__ == '__main__':
#     main(sys.argv)
#



# from collections import Counter
#
# f = open('D:\data7\\news3.txt', encoding='utf-8')
# WordDict = Counter()
# sentences = f.readlines()
#
# for sentence in sentences:
#     for word in sentence.split():
#         WordDict[word] += 1
#         # print(WordDict[word])
#         # print(word)
#         # print(WordDict)
# for word, freq in WordDict.most_common(100):
#     if word not in ('수','등','있는','것이다','있다.','것','것이다'):
#         print(word, freq)





# import os
# os.environ['_JAVA_OPTIONS'] = '-Xmx512M'
# JVM_PATH = r'C:\Program Files (x86)\Java\jdk1.8.0_111\jre\bin\server\jvm.dll'
# from konlpy.tag import Twitter
# from collections import Counter
# nlp = Twitter(JVM_PATH)
# # 단어 추출
# with open('D:\data7\\news3.txt', encoding='utf-8') as f:
#     nouns = nlp.nouns(f.read().decode('euckr'))
# # 갯수 세기
# count = Counter(nouns)
# # 상위 10개 확인하기
# print(count.most_common(10))

# from konlpy.tag import Kkma
# from konlpy.utils import pprint
# kkma = Kkma()
# k = kkma.sentences(('네, 안녕하세요. 반갑습니다.','euc-kr'))
# for x in k:
#     print(k)

""" 한겨레 신문 특정 키워드를 포함하는, 특정 날짜 이전 기사 내용 크롤러(정확도순 검색)
    python [모듈이름] [키워드] [가져올 페이지 숫자] [가져올 기사의 최근 날짜]
    [결과 파일명.txt]
    한페이지에 10개
"""

import sys
from bs4 import BeautifulSoup
import urllib.request
from urllib.parse import quote

TARGET_URL_BEFORE_KEWORD = 'http://search.hani.co.kr/Search?command=query&' \
                           'keyword='
TARGET_URL_BEFORE_UNTIL_DATE = '&media=news&sort=s&period=all&datefrom=' \
                               '2000.01.01&dateto='
TARGET_URL_REST = '&pageseq='


def get_link_from_news_title(page_num, URL, output_file):
    for i in range(page_num):
        URL_with_page_num = URL + str(i)
        source_code_from_URL = urllib.request.urlopen(URL_with_page_num)
        soup = BeautifulSoup(source_code_from_URL, 'lxml',
                             from_encoding='utf-8')
        for item in soup.select('dt > a'):
            article_URL = item['href']
            get_text(article_URL, output_file)


def get_text(URL, output_file):
    source_code_from_url = urllib.request.urlopen(URL)
    soup = BeautifulSoup(source_code_from_url, 'lxml', from_encoding='utf-8')
    content_of_article = soup.select('div.text')
    for item in content_of_article:
        string_item = str(item.find_all(text=True))
        output_file.write(string_item)


def main(argv):
    if len(sys.argv) != 5:
        print("python [모듈이름] [키워드] [가져올 페이지 숫자] "
              "[가져올 기사의 최근 날짜] [결과 파일명.txt]")
        return
    keyword = argv[1]
    page_num = int(argv[2])
    until_date = argv[3]
    output_file_name = argv[4]
    target_URL = TARGET_URL_BEFORE_KEWORD + quote(keyword) \
                 + TARGET_URL_BEFORE_UNTIL_DATE + until_date + TARGET_URL_REST
    output_file = open(output_file_name, 'w')
    get_link_from_news_title(page_num, target_URL, output_file)
    output_file.close()


if __name__ == '__main__':
    main(sys.argv)
