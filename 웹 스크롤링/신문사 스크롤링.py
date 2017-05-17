'''
- 현재 페이지의 이미지가 있는 href 링크 뽑기
'''
# import urllib.request           # 웹 브라우져에서 html 문서를 얻어오기위해 통신하기 위한 모듈
# from bs4 import BeautifulSoup  # html 문서검색 모듈
#
# def fetch_list_url():
#         list_url = "http://search.hani.co.kr/Search?command=query&keyword=%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5&media=news&sort=d&period=all&datefrom=2000.01.01&dateto=2017.05.17&pageseq=0"
#         url = urllib.request.Request(list_url) # url 요청에 따른 http 통신 헤더값을 얻어낸다
#         res = urllib.request.urlopen(url).read().decode("utf-8")
#
#         # 영어가 아닌 한글을 담아내기 위한 문자셋인 유니코드 문자셋을 사용해서 html 문서와 html  문서내의 한글을 res 변수에 담는다.
#         # 문자를 받는 set(p256)
#         # 1. US7ASCII : 영문
#         # 2. 유니코드(UTF8) : 한글, 중국어
#
#         soup = BeautifulSoup(res, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
#         soup_p = soup.find_all('p')
#         p = len(soup_p)
#
#         for i in range(p):
#             soup_p = soup.find_all('p')[i]
#             soup_p_a = soup_p.find('a')['href']
#             print(soup_p_a)
#
#         return soup_p_a
#
# fetch_list_url()



'''
- 현재 페이지의 모든 href 링크 뽑기
'''
# import urllib.request           # 웹 브라우져에서 html 문서를 얻어오기위해 통신하기 위한 모듈
# from bs4 import BeautifulSoup  # html 문서검색 모듈
#
# def fetch_list_url():
#         list_url = "http://search.hani.co.kr/Search?command=query&keyword=%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5&media=news&sort=d&period=all&datefrom=2000.01.01&dateto=2017.05.17&pageseq=0"
#         url = urllib.request.Request(list_url) # url 요청에 따른 http 통신 헤더값을 얻어낸다
#         res = urllib.request.urlopen(url).read().decode("utf-8")
#
#         # 영어가 아닌 한글을 담아내기 위한 문자셋인 유니코드 문자셋을 사용해서 html 문서와 html  문서내의 한글을 res 변수에 담는다.
#         # 문자를 받는 set(p256)
#         # 1. US7ASCII : 영문
#         # 2. 유니코드(UTF8) : 한글, 중국어
#
#         soup = BeautifulSoup(res, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
#         soup_p = soup.find_all('dt')
#         p = len(soup_p)
#
#         for i in range(p):
#             try:
#                 soup_p = soup.find_all('dt')[i]
#                 soup_p_a = soup_p.find('a')['href']
#                 print(soup_p_a)
#
#             except Exception:
#                 pass
#
#         return soup_p_a
#
# fetch_list_url()



'''
- 1~20 페이지의 모든 href 링크 뽑기
'''
# import urllib.request           # 웹 브라우져에서 html 문서를 얻어오기위해 통신하기 위한 모듈
# from bs4 import BeautifulSoup  # html 문서검색 모듈
#
# def fetch_list_url():
#     for i in range(0,20):
#         list_url = 'http://search.hani.co.kr/Search?command=query&keyword=%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5&media=news&sort=d&period=all&datefrom=2000.01.01&dateto=2017.05.17&pageseq=0'+str(i)
#         url = urllib.request.Request(list_url) # url 요청에 따른 http 통신 헤더값을 얻어낸다
#         res = urllib.request.urlopen(url).read().decode("utf-8")
#
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
#                 print(soup_p_a)
#
#             except Exception:
#                 pass
#
# fetch_list_url()


'''
- url 담기
'''
# import urllib.request           # 웹 브라우져에서 html 문서를 얻어오기위해 통신하기 위한 모듈
# from bs4 import BeautifulSoup  # html 문서검색 모듈
#
# def fetch_list_url():
#     params = []
#     for i in range(0,20):
#         list_url = 'http://search.hani.co.kr/Search?command=query&keyword=%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5&media=news&sort=d&period=all&datefrom=2000.01.01&dateto=2017.05.17&pageseq=0'+str(i)
#         url = urllib.request.Request(list_url) # url 요청에 따른 http 통신 헤더값을 얻어낸다
#         res = urllib.request.urlopen(url).read().decode("utf-8")
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
#     return params
#
# fetch_list_url()




'''
- 해당 url의 상세기사 출력
'''
# import urllib.request  # 웹 브라우져에서 html 문서를 얻어오기위해 통신하기 위한 모듈
# from bs4 import BeautifulSoup  # html 문서검색 모듈
#
# def fetch_list_url2():
#     list_url = "http://www.hani.co.kr/arti/culture/book/778730.html"
#     url = urllib.request.Request(list_url)  # url 요청에 따른 http 통신 헤더값을 얻어낸다
#     res = urllib.request.urlopen(url).read().decode("utf-8")
#
#     # 영어가 아닌 한글을 담아내기 위한 문자셋인 유니코드 문자셋을 사용해서 html 문서와 html  문서내의 한글을 res 변수에 담는다.
#     # 문자를 받는 set(p256)
#     # 1. US7ASCII : 영문
#     # 2. 유니코드(UTF8) : 한글, 중국어
#
#     soup = BeautifulSoup(res, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
#     soup_p = soup.find_all('div', class_='article-text')
#
#     print(soup_p[0].get_text())
#
#
# fetch_list_url2()




'''
- 1~20페이지의 모든 상세기사 출력
'''
# import urllib.request           # 웹 브라우져에서 html 문서를 얻어오기위해 통신하기 위한 모듈
# from bs4 import BeautifulSoup  # html 문서검색 모듈
#
# def fetch_list_url():
#     params = []
#     for i in range(0,20):
#         list_url = 'http://search.hani.co.kr/Search?command=query&keyword=%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5&media=news&sort=d&period=all&datefrom=2000.01.01&dateto=2017.05.17&pageseq='+str(i)
#         url = urllib.request.Request(list_url) # url 요청에 따른 http 통신 헤더값을 얻어낸다
#         res = urllib.request.urlopen(url).read().decode("utf-8")
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
#         # print(params)
#     return params
#
# def fetch_list_url2():
#     params2 = fetch_list_url()
#     # print(params2)
#     for i in params2:
#         list_url2 = i
#         url2 = urllib.request.Request(list_url2)  # url 요청에 따른 http 통신 헤더값을 얻어낸다
#         res2 = urllib.request.urlopen(url2).read().decode("utf-8")
#         soup = BeautifulSoup(res2, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
#         soup_p = soup.find_all('div', class_='article-text')[0]
#
#         print(soup_p.get_text(strip=True))
#
# fetch_list_url2()



"""
- 한겨래 txt 파일로 기사 담기
"""
import urllib.request           # 웹 브라우져에서 html 문서를 얻어오기위해 통신하기 위한 모듈
from bs4 import BeautifulSoup  # html 문서검색 모듈
import os

def get_save_path():
    save_path = input("Enter the file name and file location : ")
    save_path = save_path.replace("\\", "/")

    if not os.path.isdir(os.path.split(save_path)[0]):
        os.mkdir(os.path.split(save_path)[0])   # 폴더가 없으면 만드는 작업
    return save_path

def fetch_list_url():
    params = []
    for i in range(0,20):
        list_url = 'http://search.hani.co.kr/Search?command=query&keyword=%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5&media=news&sort=d&period=all&datefrom=2000.01.01&dateto=2017.05.17&pageseq='+str(i)
        url = urllib.request.Request(list_url) # url 요청에 따른 http 통신 헤더값을 얻어낸다
        res = urllib.request.urlopen(url).read().decode("utf-8")
        soup = BeautifulSoup(res, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
        soup_p = soup.find_all('dt')
        p = len(soup_p)

        for j in range(p):
            try:
                soup_p = soup.find_all('dt')[j]
                soup_p_a = soup_p.find('a')['href']
                params.append(soup_p_a)

            except Exception:
                pass
        # print(params)
    return params

def fetch_list_url2():
    params2 = fetch_list_url()
    # print(params2)
    f = open(get_save_path(), 'w', encoding='utf-8')

    for i in params2:
        list_url2 = i
        url2 = urllib.request.Request(list_url2)  # url 요청에 따른 http 통신 헤더값을 얻어낸다
        res2 = urllib.request.urlopen(url2).read().decode("utf-8")
        soup = BeautifulSoup(res2, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
        soup_p = soup.find_all('div', class_='article-text')[0]

        # print(soup_p.get_text(strip=True))
        result = soup_p.get_text(strip=True, separator='\n')
        f.write(result+'\n')

    f.close()

fetch_list_url2()






'''
- 중앙일보 
'''
import urllib.request           # 웹 브라우져에서 html 문서를 얻어오기위해 통신하기 위한 모듈
from bs4 import BeautifulSoup  # html 문서검색 모듈
import os

def get_save_path():
    save_path = input("Enter the file name and file location : ")
    save_path = save_path.replace("\\", "/")

    if not os.path.isdir(os.path.split(save_path)[0]):
        os.mkdir(os.path.split(save_path)[0])   # 폴더가 없으면 만드는 작업
    return save_path


def fetch_list_url():
    params = []
    for i in range(1,20):
        list_url = 'http://search.joins.com/JoongangNews?page='+str(i)+'&Keyword=%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5&SortType=New&SearchCategoryType=JoongangNews&MatchKeyword=%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5'
        url = urllib.request.Request(list_url) # url 요청에 따른 http 통신 헤더값을 얻어낸다
        res = urllib.request.urlopen(url).read().decode("utf-8")
        soup = BeautifulSoup(res, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
        soup_p = soup.find_all('div', class_="text")
        p = len(soup_p)

        for j in range(p):
            try:
                soup_p = soup.find_all('div', class_="text")[j]
                soup_p_a = soup_p.find('a')['href']
                params.append(soup_p_a)

            except Exception:
                pass

    return params

def fetch_list_url2():
    params2 = fetch_list_url()
    # print(params2)
    f = open(get_save_path(), 'w', encoding='utf-8')

    for i in params2:
        list_url2 = i
        url2 = urllib.request.Request(list_url2)  # url 요청에 따른 http 통신 헤더값을 얻어낸다
        res2 = urllib.request.urlopen(url2).read().decode("utf-8")
        soup = BeautifulSoup(res2, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
        soup_p = soup.find_all('div', id="article_body")[0]

        # print(soup_p.get_text(strip=True))
        result = soup_p.get_text(strip=True, separator='\n')
        f.write(result+'\n')

    f.close()

fetch_list_url2()