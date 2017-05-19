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
# import urllib.request           # 웹 브라우져에서 html 문서를 얻어오기위해 통신하기 위한 모듈
# from bs4 import BeautifulSoup  # html 문서검색 모듈
# import os
#
# def get_save_path():
#     save_path = input("Enter the file name and file location : ")
#     save_path = save_path.replace("\\", "/")
#
#     if not os.path.isdir(os.path.split(save_path)[0]):
#         os.mkdir(os.path.split(save_path)[0])   # 폴더가 없으면 만드는 작업
#     return save_path
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
#     f = open(get_save_path(), 'w', encoding='utf-8')
#
#     for i in params2:
#         list_url2 = i
#         url2 = urllib.request.Request(list_url2)  # url 요청에 따른 http 통신 헤더값을 얻어낸다
#         res2 = urllib.request.urlopen(url2).read().decode("utf-8")
#         soup = BeautifulSoup(res2, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
#         soup_p = soup.find_all('div', class_='article-text')[0]
#
#         # print(soup_p.get_text(strip=True))
#         result = soup_p.get_text(strip=True, separator='\n')
#         f.write(result+'\n')
#
#     f.close()
#
# fetch_list_url2()






'''
- 중앙일보 
'''
# import urllib.request           # 웹 브라우져에서 html 문서를 얻어오기위해 통신하기 위한 모듈
# from bs4 import BeautifulSoup  # html 문서검색 모듈
# import os
#
# def get_save_path():
#     save_path = input("Enter the file name and file location : ")
#     save_path = save_path.replace("\\", "/")
#
#     if not os.path.isdir(os.path.split(save_path)[0]):
#         os.mkdir(os.path.split(save_path)[0])   # 폴더가 없으면 만드는 작업
#     return save_path
#
#
# def fetch_list_url():
#     params = []
#     for i in range(1,20):
#         list_url = 'http://search.joins.com/JoongangNews?page='+str(i)+'&Keyword=%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5&SortType=New&SearchCategoryType=JoongangNews&MatchKeyword=%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5'
#         url = urllib.request.Request(list_url) # url 요청에 따른 http 통신 헤더값을 얻어낸다
#         res = urllib.request.urlopen(url).read().decode("utf-8")
#         soup = BeautifulSoup(res, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
#         soup_p = soup.find_all('div', class_="text")
#         p = len(soup_p)
#
#         for j in range(p):
#             try:
#                 soup_p = soup.find_all('div', class_="text")[j]
#                 soup_p_a = soup_p.find('a')['href']
#                 params.append(soup_p_a)
#
#             except Exception:
#                 pass
#
#     return params
#
# def fetch_list_url2():
#     params2 = fetch_list_url()
#     # print(params2)
#     f = open(get_save_path(), 'w', encoding='utf-8')
#
#     for i in params2:
#         list_url2 = i
#         url2 = urllib.request.Request(list_url2)  # url 요청에 따른 http 통신 헤더값을 얻어낸다
#         res2 = urllib.request.urlopen(url2).read().decode("utf-8")
#         soup = BeautifulSoup(res2, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
#         soup_p = soup.find_all('div', id="article_body")[0]
#
#         # print(soup_p.get_text(strip=True))
#         result = soup_p.get_text(strip=True, separator='\n')
#         f.write(result+'\n')
#
#     f.close()
#
# fetch_list_url2()




'''
- 서울시 응답소 게시판의 웹페이지의 html 문서를 스크롤링 하기 위한 메인 페이지의 html 가져오기
'''
# import urllib.request           # 웹 브라우져에서 html 문서를 얻어오기위해 통신하기 위한 모듈
# from bs4 import BeautifulSoup  # html 문서검색 모듈
#
# def fetch_list_url():
#     # params = []
#     # for i in range(1,30):
#         list_url = 'http://eungdapso.seoul.go.kr/Shr/Shr01/Shr01_lis.jsp'
#
#         request_header = urllib.parse.urlencode({'page':1})
#         # print(request_header) # 결과 page=1, page=2, .......
#
#         request_header = request_header.encode("utf-8")
#         # print(request_header)   # 결과 b'page=1', b'page=2',.......
#
#         url = urllib.request.Request(list_url, request_header)  # url 요청에 따른 http 통신 헤더값을 얻어낸다
#         # print(url)
#
#         res = urllib.request.urlopen(url).read().decode("utf-8")
#         print(res)
#
# fetch_list_url()




'''
- 1 페이지 html 문서중에 li 태그에 class="pclist_list_tit2" 검색!
'''
# import urllib.request  # 웹 브라우져에서 html 문서를 얻어오기위해 통신하기 위한 모듈
# from bs4 import BeautifulSoup  # html 문서검색 모듈
#
#
# def fetch_list_url():
#     # params = []
#     # for i in range(1,30):
#         list_url = 'http://eungdapso.seoul.go.kr/Shr/Shr01/Shr01_lis.jsp'
#
#         request_header = urllib.parse.urlencode({'page': 1})
#         # print(request_header) # 결과 page=1, page=2, .......
#
#         request_header = request_header.encode("utf-8")
#         # print(request_header)   # 결과 b'page=1', b'page=2',.......
#
#         url = urllib.request.Request(list_url, request_header)  # url 요청에 따른 http 통신 헤더값을 얻어낸다
#         # print(url)
#
#         res = urllib.request.urlopen(url).read().decode("utf-8")
#         # print(res)
#
#         soup = BeautifulSoup(res, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
#         soup_p = soup.find_all('li', class_="pclist_list_tit2")
#         print(soup_p)
#
# fetch_list_url()




'''
- a 태그의 href에 해당하는 부분만 검색!
'''
# import urllib.request  # 웹 브라우져에서 html 문서를 얻어오기위해 통신하기 위한 모듈
# from bs4 import BeautifulSoup  # html 문서검색 모듈
#
#
# def fetch_list_url():
#     # params = []
#     # for i in range(1,30):
#         list_url = 'http://eungdapso.seoul.go.kr/Shr/Shr01/Shr01_lis.jsp'
#
#         request_header = urllib.parse.urlencode({'page': 1})
#         # print(request_header) # 결과 page=1, page=2, .......
#
#         request_header = request_header.encode("utf-8")
#         # print(request_header)   # 결과 b'page=1', b'page=2',.......
#
#         url = urllib.request.Request(list_url, request_header)  # url 요청에 따른 http 통신 헤더값을 얻어낸다
#         # print(url)
#
#         res = urllib.request.urlopen(url).read().decode("utf-8")
#         # print(res)
#
#         soup = BeautifulSoup(res, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
#         soup_p = soup.find_all('li', class_="pclist_list_tit2")
#         # print(soup_p)
#
#         soup_p_a = soup_p[0].find('a')['href']
#         print(soup_p_a)
#
# fetch_list_url()



'''
- 정규식 모듈인 re 를 이용해서 위의 결과에서 숫자만 출력!
'''
# import urllib.request  # 웹 브라우져에서 html 문서를 얻어오기위해 통신하기 위한 모듈
# from bs4 import BeautifulSoup  # html 문서검색 모듈
# import re
#
# def fetch_list_url():
#     # params = []
#     # for i in range(1,30):
#         list_url = 'http://eungdapso.seoul.go.kr/Shr/Shr01/Shr01_lis.jsp'
#
#         request_header = urllib.parse.urlencode({'page': 1})
#         # print(request_header) # 결과 page=1, page=2, .......
#
#         request_header = request_header.encode("utf-8")
#         # print(request_header)   # 결과 b'page=1', b'page=2',.......
#
#         url = urllib.request.Request(list_url, request_header)  # url 요청에 따른 http 통신 헤더값을 얻어낸다
#         # print(url)
#
#         res = urllib.request.urlopen(url).read().decode("utf-8")
#         # print(res)
#
#         soup = BeautifulSoup(res, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
#         soup_p = soup.find_all('li', class_="pclist_list_tit2")
#         # print(soup_p)
#
#         soup_p_a = soup_p[0].find('a')['href']
#         # print(soup_p_a)
#
#         print(re.search('[0-9]{14}', soup_p_a).group())
#
# fetch_list_url()



'''
- 현재 페이지의 모든 href의 숫자 출력
'''
# import urllib.request  # 웹 브라우져에서 html 문서를 얻어오기위해 통신하기 위한 모듈
# from bs4 import BeautifulSoup  # html 문서검색 모듈
# import re
#
# def fetch_list_url():
#     # params = []
#     # for i in range(1,30):
#         list_url = 'http://eungdapso.seoul.go.kr/Shr/Shr01/Shr01_lis.jsp'
#
#         request_header = urllib.parse.urlencode({'page': 1})
#         # print(request_header) # 결과 page=1, page=2, .......
#
#         request_header = request_header.encode("utf-8")
#         # print(request_header)   # 결과 b'page=1', b'page=2',.......
#
#         url = urllib.request.Request(list_url, request_header)  # url 요청에 따른 http 통신 헤더값을 얻어낸다
#         # print(url)
#
#         res = urllib.request.urlopen(url).read().decode("utf-8")
#         # print(res)
#
#         soup = BeautifulSoup(res, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
#         soup_p = soup.find_all('li', class_="pclist_list_tit2")
#         # print(soup_p)
#
#         soup_p_a = soup_p[0].find('a')['href']
#         # print(soup_p_a)
#
#         # print(re.search('[0-9]{14}', soup_p_a).group())
#
#         p = len(soup_p)
#
#         for j in range(p):
#             soup_p = soup.find_all('li', class_="pclist_list_tit2")[j]
#             soup_p_a = soup_p.find('a')['href']
#             print(re.search('[0-9]{14}', soup_p_a).group())
#
# fetch_list_url()



'''
- 게시 페이지 번호 1번부터 30번까지 href 숫자 출력!
'''
# import urllib.request  # 웹 브라우져에서 html 문서를 얻어오기위해 통신하기 위한 모듈
# from bs4 import BeautifulSoup  # html 문서검색 모듈
# import re
#
# def fetch_list_url():
#     # params = []
#     for i in range(1,30):
#         list_url = 'http://eungdapso.seoul.go.kr/Shr/Shr01/Shr01_lis.jsp'
#
#         request_header = urllib.parse.urlencode({'page': i})
#         # print(request_header) # 결과 page=1, page=2, .......
#
#         request_header = request_header.encode("utf-8")
#         # print(request_header)   # 결과 b'page=1', b'page=2',.......
#
#         url = urllib.request.Request(list_url, request_header)  # url 요청에 따른 http 통신 헤더값을 얻어낸다
#         # print(url)
#
#         res = urllib.request.urlopen(url).read().decode("utf-8")
#         # print(res)
#
#         soup = BeautifulSoup(res, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
#         soup_p = soup.find_all('li', class_="pclist_list_tit2")
#         # print(soup_p)
#
#         soup_p_a = soup_p[0].find('a')['href']
#         # print(soup_p_a)
#
#         # print(re.search('[0-9]{14}', soup_p_a).group())
#
#         p = len(soup_p)
#
#         for j in range(p):
#             soup_p = soup.find_all('li', class_="pclist_list_tit2")[j]
#             soup_p_a = soup_p.find('a')['href']
#             print(re.search('[0-9]{14}', soup_p_a).group())
#
# fetch_list_url()



'''
- params 라는 비어있는 리스트 변수에 위의 결과를 담고 리턴
'''
# import urllib.request  # 웹 브라우져에서 html 문서를 얻어오기위해 통신하기 위한 모듈
# from bs4 import BeautifulSoup  # html 문서검색 모듈
# import re
#
# def fetch_list_url():
#     params = []
#     for i in range(1,30):
#         list_url = 'http://eungdapso.seoul.go.kr/Shr/Shr01/Shr01_lis.jsp'
#
#         request_header = urllib.parse.urlencode({'page': i})
#         # print(request_header) # 결과 page=1, page=2, .......
#
#         request_header = request_header.encode("utf-8")
#         # print(request_header)   # 결과 b'page=1', b'page=2',.......
#
#         url = urllib.request.Request(list_url, request_header)  # url 요청에 따른 http 통신 헤더값을 얻어낸다
#         # print(url)
#
#         res = urllib.request.urlopen(url).read().decode("utf-8")
#         # print(res)
#
#         soup = BeautifulSoup(res, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
#         soup_p = soup.find_all('li', class_="pclist_list_tit2")
#         # print(soup_p)
#
#         soup_p_a = soup_p[0].find('a')['href']
#         # print(soup_p_a)
#
#         # print(re.search('[0-9]{14}', soup_p_a).group())
#
#         p = len(soup_p)
#
#         for j in range(p):
#             soup_p = soup.find_all('li', class_="pclist_list_tit2")[j]
#             soup_p_a = soup_p.find('a')['href']
#             params.append(re.search('[0-9]{14}', soup_p_a).group())
#
#     print(params)
#     return params
#
# fetch_list_url()



'''
- 게시글에 대한 href 숫자 20170504003012를 가지고 상세 게시판 글을 스크롤링 하는fetch_list_url2() 함수 생성
'''
# import urllib.request  # 웹 브라우져에서 html 문서를 얻어오기위해 통신하기 위한 모듈
# from bs4 import BeautifulSoup  # html 문서검색 모듈
#
# def fetch_list_url2():
#     detail_url = 'http://eungdapso.seoul.go.kr/Shr/Shr01/Shr01_vie.jsp'
#
#     request_header = urllib.parse.urlencode({'RCEPT_NO': '20170504003012'})
#     request_header = request_header.encode("utf-8")
#
#     url = urllib.request.Request(detail_url, request_header)  # url 요청에 따른 http 통신 헤더값을 얻어낸다
#     res = urllib.request.urlopen(url).read().decode("utf-8")
#
#     soup = BeautifulSoup(res, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
#     print(soup)
#
# fetch_list_url2()




'''
- 위의 html 문서에서 div 태그의 form_table 클래스부분 검색
'''
# import urllib.request  # 웹 브라우져에서 html 문서를 얻어오기위해 통신하기 위한 모듈
# from bs4 import BeautifulSoup  # html 문서검색 모듈
#
# def fetch_list_url2():
#     detail_url = 'http://eungdapso.seoul.go.kr/Shr/Shr01/Shr01_vie.jsp'
#
#     request_header = urllib.parse.urlencode({'RCEPT_NO': '20170504003012'})
#     request_header = request_header.encode("utf-8")
#
#     url = urllib.request.Request(detail_url, request_header)  # url 요청에 따른 http 통신 헤더값을 얻어낸다
#     res = urllib.request.urlopen(url).read().decode("utf-8")
#
#     soup = BeautifulSoup(res, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
#     # print(soup)
#     soup_p = soup.find('div', class_="form_table")
#     print(soup_p)
#
# fetch_list_url2()



'''
- 위에서 form_table 태그에 해당하는 html 문서를 가져왔는데 그 문서 안에 table 태그에 해당하는 html 문서를 모두 검색!
'''
# import urllib.request  # 웹 브라우져에서 html 문서를 얻어오기위해 통신하기 위한 모듈
# from bs4 import BeautifulSoup  # html 문서검색 모듈
#
# def fetch_list_url2():
#     detail_url = 'http://eungdapso.seoul.go.kr/Shr/Shr01/Shr01_vie.jsp'
#
#     request_header = urllib.parse.urlencode({'RCEPT_NO': '20170504003012'})
#     request_header = request_header.encode("utf-8")
#
#     url = urllib.request.Request(detail_url, request_header)  # url 요청에 따른 http 통신 헤더값을 얻어낸다
#     res = urllib.request.urlopen(url).read().decode("utf-8")
#
#     soup = BeautifulSoup(res, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
#     # print(soup)
#     soup_p = soup.find('div', class_="form_table")
#     # print(soup_p)
#
#     soup_p_a = soup_p.find_all('table')
#     print(soup_p_a)
#
# fetch_list_url2()






'''
- 게시날짜 출력
'''
# import urllib.request  # 웹 브라우져에서 html 문서를 얻어오기위해 통신하기 위한 모듈
# from bs4 import BeautifulSoup  # html 문서검색 모듈
#
# def fetch_list_url2():
#     detail_url = 'http://eungdapso.seoul.go.kr/Shr/Shr01/Shr01_vie.jsp'
#
#     request_header = urllib.parse.urlencode({'RCEPT_NO': '20170504003012'})
#     request_header = request_header.encode("utf-8")
#
#     url = urllib.request.Request(detail_url, request_header)  # url 요청에 따른 http 통신 헤더값을 얻어낸다
#     res = urllib.request.urlopen(url).read().decode("utf-8")
#
#     soup = BeautifulSoup(res, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
#     # print(soup)
#     soup_p = soup.find('div', class_="form_table")
#     # print(soup_p)
#
#     tables = soup_p.find_all('table')
#     date = tables[0].find_all('td')
#     date2 = date[1].get_text()
#     print(date2)
#
# fetch_list_url2()




'''
- 제목 출력
'''
# import urllib.request  # 웹 브라우져에서 html 문서를 얻어오기위해 통신하기 위한 모듈
# from bs4 import BeautifulSoup  # html 문서검색 모듈
#
# def fetch_list_url2():
#     detail_url = 'http://eungdapso.seoul.go.kr/Shr/Shr01/Shr01_vie.jsp'
#
#     request_header = urllib.parse.urlencode({'RCEPT_NO': '20170504003012'})
#     request_header = request_header.encode("utf-8")
#
#     url = urllib.request.Request(detail_url, request_header)  # url 요청에 따른 http 통신 헤더값을 얻어낸다
#     res = urllib.request.urlopen(url).read().decode("utf-8")
#
#     soup = BeautifulSoup(res, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
#     # print(soup)
#     soup_p = soup.find('div', class_="form_table")
#     # print(soup_p)
#
#     tables = soup_p.find_all('table')
#     table0 = tables[0].find_all('td')
#     date = table0[1].get_text()
#     title = table0[0].get_text()
#     print(title)
#
# fetch_list_url2()



'''
- 민원 내용을 question 변수에 담고 출력
'''
# import urllib.request  # 웹 브라우져에서 html 문서를 얻어오기위해 통신하기 위한 모듈
# from bs4 import BeautifulSoup  # html 문서검색 모듈
#
# def fetch_list_url2():
#     detail_url = 'http://eungdapso.seoul.go.kr/Shr/Shr01/Shr01_vie.jsp'
#
#     request_header = urllib.parse.urlencode({'RCEPT_NO': '20170504003012'})
#     request_header = request_header.encode("utf-8")
#
#     url = urllib.request.Request(detail_url, request_header)  # url 요청에 따른 http 통신 헤더값을 얻어낸다
#     res = urllib.request.urlopen(url).read().decode("utf-8")
#
#     soup = BeautifulSoup(res, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
#     # print(soup)
#     soup_p = soup.find('div', class_="form_table")
#     # print(soup_p)
#
#     tables = soup_p.find_all('table')
#     table0 = tables[0].find_all('td')
#     table1 = tables[1].find('div', class_="table_inner_desc")
#
#     date = table0[1].get_text()
#     title = table0[0].get_text()
#     question = table1.get_text(strip=True)
#
#     print(question)
#
# fetch_list_url2()


'''
- 답변을 answer 변수에 담고 출력
'''
# import urllib.request  # 웹 브라우져에서 html 문서를 얻어오기위해 통신하기 위한 모듈
# from bs4 import BeautifulSoup  # html 문서검색 모듈
#
# def fetch_list_url2():
#     detail_url = 'http://eungdapso.seoul.go.kr/Shr/Shr01/Shr01_vie.jsp'
#
#     request_header = urllib.parse.urlencode({'RCEPT_NO': '20170504003012'})
#     request_header = request_header.encode("utf-8")
#
#     url = urllib.request.Request(detail_url, request_header)  # url 요청에 따른 http 통신 헤더값을 얻어낸다
#     res = urllib.request.urlopen(url).read().decode("utf-8")
#
#     soup = BeautifulSoup(res, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
#     # print(soup)
#     soup_p = soup.find('div', class_="form_table")
#     # print(soup_p)
#
#     tables = soup_p.find_all('table')
#     table0 = tables[0].find_all('td')
#     table1 = tables[1].find('div', class_="table_inner_desc")
#     table2 = tables[2].find('div', class_='table_inner_desc')
#
#     date = table0[1].get_text()
#     title = table0[0].get_text()
#     question = table1.get_text(strip=True)
#     answer = table2.get_text(strip=True)
#     print(answer)
#
# fetch_list_url2()




# import urllib.request  # 웹브라우저에서 html 문서를 얻어오기위해 통신하는 모듈
# from  bs4 import BeautifulSoup  # html 문서 검색 모듈
# import os
# import re
#
#
# def fetch_list_url():
#     params = []
#     for j in range(1, 30):
#
#         list_url = "http://eungdapso.seoul.go.kr/Shr/Shr01/Shr01_lis.jsp"
#
#         request_header = urllib.parse.urlencode({"page": j})
#         # print (request_header) # 결과 page=1, page=2 ..
#
#         request_header = request_header.encode("utf-8")
#         # print (request_header) # b'page=29'
#
#         url = urllib.request.Request(list_url, request_header)
#         # print (url) # <urllib.request.Request object at 0x00000000021FA2E8>
#
#         res = urllib.request.urlopen(url).read().decode("utf-8")
#
#         soup = BeautifulSoup(res, "html.parser")
#         soup2 = soup.find_all("li", class_="pclist_list_tit2")
#         for soup3 in soup2:
#             soup4 = soup3.find("a")["href"]
#             params.append(re.search("[0-9]{14}", soup4).group())
#
#
#     return params
#
#
# def fetch_list_url2():
#
#     params2 = fetch_list_url()
#     for i in params2:
#
#         detail_url = "http://eungdapso.seoul.go.kr/Shr/Shr01/Shr01_vie.jsp"
#
#         request_header = urllib.parse.urlencode({"RCEPT_NO": str(i) })
#         request_header = request_header.encode("utf-8")
#
#         url = urllib.request.Request(detail_url, request_header)
#         res = urllib.request.urlopen(url).read().decode("utf-8")
#         soup = BeautifulSoup(res, "html.parser")
#         soup2 = soup.find("div", class_="form_table")
#
#         tables = soup2.find_all("table")
#         table0   = tables[0].find_all("td")
#         table1   = tables[1].find("div",class_="table_inner_desc")
#         table2   = tables[2].find("div",class_="table_inner_desc")
#
#         date  = table0[1].get_text()
#         title = table0[0].get_text()
#         question = table1.get_text(strip=True)
#         answer   = table2.get_text(strip=True)
#         print(answer)
#
# fetch_list_url2()



'''
- 서울응답소 txt 파일생성
'''
# import urllib.request  # 웹브라우저에서 html 문서를 얻어오기위해 통신하는 모듈
# from  bs4 import BeautifulSoup  # html 문서 검색 모듈
# import os
# import re
#
# def get_save_path():
#     save_path = input("Enter the file name and file location : ")
#     save_path = save_path.replace("\\", "/")
#
#     if not os.path.isdir(os.path.split(save_path)[0]):
#         os.mkdir(os.path.split(save_path)[0])   # 폴더가 없으면 만드는 작업
#     return save_path
#
# def fetch_list_url():
#     params = []
#     for j in range(1, 30):
#
#         list_url = "http://eungdapso.seoul.go.kr/Shr/Shr01/Shr01_lis.jsp"
#
#         request_header = urllib.parse.urlencode({"page": j})
#         # print (request_header) # 결과 page=1, page=2 ..
#
#         request_header = request_header.encode("utf-8")
#         # print (request_header) # b'page=29'
#
#         url = urllib.request.Request(list_url, request_header)
#         # print (url) # <urllib.request.Request object at 0x00000000021FA2E8>
#
#         res = urllib.request.urlopen(url).read().decode("utf-8")
#
#         soup = BeautifulSoup(res, "html.parser")
#         soup2 = soup.find_all("li", class_="pclist_list_tit2")
#         for soup3 in soup2:
#             soup4 = soup3.find("a")["href"]
#             params.append(re.search("[0-9]{14}", soup4).group())
#
#
#     return params
#
#
# def fetch_list_url2():
#
#     params2 = fetch_list_url()
#     f = open(get_save_path(), 'w', encoding='utf-8')
#     for i in params2:
#
#         detail_url = "http://eungdapso.seoul.go.kr/Shr/Shr01/Shr01_vie.jsp"
#
#         request_header = urllib.parse.urlencode({"RCEPT_NO": str(i) })
#         request_header = request_header.encode("utf-8")
#
#         url = urllib.request.Request(detail_url, request_header)
#         res = urllib.request.urlopen(url).read().decode("utf-8")
#         soup = BeautifulSoup(res, "html.parser")
#         soup2 = soup.find("div", class_="form_table")
#
#         tables = soup2.find_all("table")
#         table0 = tables[0].find_all("td")
#         table1 = tables[1].find("div",class_="table_inner_desc")
#         table2 = tables[2].find("div",class_="table_inner_desc")
#
#         date = table0[1].get_text()
#         title = table0[0].get_text()
#         question = table1.get_text(strip=True)
#         answer = table2.get_text(strip=True)
#         # print(answer)
#         f.write('=='*30 + '\n')
#         f.write(title + '\n')
#         f.write(date + '\n')
#         f.write(question + '\n')
#         f.write(answer + '\n')
#         f.write('=='*30 + '\n')
#
#     f.close()
#
# fetch_list_url2()















import urllib.request
from  bs4 import BeautifulSoup
from selenium import webdriver      # 웹 어플리케이션의 테스트를 자동화하기 위한 프레임 워크
from selenium.webdriver.common.keys import Keys
import time         # 중간중간 sleep을 걸어야해서 time 모듈을 import 한다

binary = 'D:\chromedriver/chromedriver.exe'
# 웹 브라우저를 크롬을 사용할거라서 크롬 드라이버를 다운받아서 위의 위치에 둔다
# 팬텀 js 로 하면 백 그라운드로 실행할 수 있다

browser = webdriver.Chrome(binary)  # 브라우저를 인스턴스화
browser.get("https://www.google.co.kr/imghp?hl=")
# 네이버의 이미지 검색 url 을 받아온다

elem = browser.find_element_by_id("lst-ib")
# 네이버의 이미지 검색에 해당하는 input 창의 id 가 nx_query여서 검색창의 해당하는 hteml 코드를
# 찾아서 eleml요소 사용하겠금 설정

# find_elements_by_class_name("") # 클래스 이름을 찾을때 방법

# 검색어 입력
elem.send_keys("장미꽃")  # elem 이 input 창과 연결이 되어서 스스로 아이언맨을 검색창에 쓴다
elem.submit()              # 웹에서의 submit 은 엔터의 역할

# 반복할 횟수
for i in range(1,2):   # 2번만 누르게 한다
    browser.find_element_by_xpath("//body").send_keys(Keys.END)
    # 브라우저 아무데서나 end 키를 누른다고 해서 페이지가 아래로 내려가지 않아서 body 활성화시킨후 end 키를 누른다

    time.sleep(5)   # end 로 내려오는데 시간이 걸려서 5초의 sleep

time.sleep(5)       # 네트워크 안정성(느릴까봐) 5초의 sleep

html = browser.page_source  # 크롬 브라우저에서 현재 불러온 소스를 가져온다

soup = BeautifulSoup(html, "lxml")  # BeautifulSoup 를 사용해서 html 코드를 검색할 수 있도록 설정

# print(soup)
# print(len(soup))


def fetch_list_url():
    params = []
    imgList = soup.find_all("img", class_="rg_ic rg_i")   # 네이버 이미지 url 이 있는 img 태그의 _img 클래스로 가서
    for im in imgList:
        params.append(im["src"])                    # params 리스트 변수에 image_url 을 담는다
    return params


def fetch_detail_url():
    params = fetch_list_url()
    print(params)
    a = 1
    for p in params:
        print(p)
        # 다운받을 폴더경로 입력
        urllib.request.urlretrieve(p, "d:/GoogleImage/" + str(a) + ".jpg")
        a += 1

fetch_detail_url()

browser.quit()