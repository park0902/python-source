# from bs4 import BeautifulSoup
#
# with open('d:\\a.html') as kimnamhoon:
#     soup = BeautifulSoup(kimnamhoon, 'lxml')    # BeautifulSoup 클래스의 매개변수를
#                                                 # 두 개를 사용하고 soup를 인스턴스화
# print(soup.title.string)


# from bs4 import BeautifulSoup
#
# with open('d:\\a.html') as kimnamhoon:
#     soup = BeautifulSoup(kimnamhoon, 'lxml')    # BeautifulSoup 클래스의 매개변수를
#                                                 # 두 개를 사용하고 soup를 인스턴스화
# print(soup.find('a'))                           # a 태그 첫번째 요소만 출력
# print(soup.find_all('a'))                       # a 태그 모든 요소 출력'''



# from bs4 import BeautifulSoup
#
# with open('d:\\a.html') as kimnamhoon:
#     soup = BeautifulSoup(kimnamhoon, 'lxml')    # BeautifulSoup 클래스의 매개변수를
#                                                 # 두 개를 사용하고 soup를 인스턴스화
# for link in soup.find_all('a'):
#     print(link)
#     print(link.get('href'))



# from bs4 import BeautifulSoup
#
# with open('d:\\a.html') as kimnamhoon:
#     soup = BeautifulSoup(kimnamhoon, 'lxml')    # BeautifulSoup 클래스의 매개변수를
#                                                 # 두 개를 사용하고 soup를 인스턴스화
# print(soup.get_text())



# from bs4 import BeautifulSoup
#
# with open('d:\\a.html') as kimnamhoon:
#     soup = BeautifulSoup(kimnamhoon, 'lxml')    # BeautifulSoup 클래스의 매개변수를
#                                                 # 두 개를 사용하고 soup를 인스턴스화
# print(soup.get_text(strip=True))




# from bs4 import BeautifulSoup
#
# with open('d:\data\ecologicalpyramid.html') as kimnamhoon:
#     soup = BeautifulSoup(kimnamhoon, 'lxml')    # BeautifulSoup 클래스의 매개변수를
#                                                 # 두 개를 사용하고 soup를 인스턴스화
# result = soup.find(class_='number')
# print(result)
# print(result.get_text())



# from bs4 import BeautifulSoup
#
# with open('d:\data\ecologicalpyramid.html') as kimnamhoon:
#     soup = BeautifulSoup(kimnamhoon, 'lxml')    # BeautifulSoup 클래스의 매개변수를
#                                                 # 두 개를 사용하고 soup를 인스턴스화
# result = soup.find_all(class_='number')
#
# for link in result:
#     print(link.get_text())



# from bs4 import BeautifulSoup
#
# with open('d:\data\ecologicalpyramid.html') as kimnamhoon:
#     soup = BeautifulSoup(kimnamhoon, 'lxml')    # BeautifulSoup 클래스의 매개변수를
#                                                 # 두 개를 사용하고 soup를 인스턴스화
# result = soup.find_all(class_='number')[2]
#
# print(result.get_text())



# from bs4 import BeautifulSoup
#
# with open('d:\data\ecologicalpyramid.html') as kimnamhoon:
#     soup = BeautifulSoup(kimnamhoon, 'lxml')    # BeautifulSoup 클래스의 매개변수를
#                                                 # 두 개를 사용하고 soup를 인스턴스화
# result = soup.find_all(class_='name')[4]
#
# print(result.get_text())




# from bs4 import BeautifulSoup
#
# with open('d:\data\ecologicalpyramid.html') as kimnamhoon:
#     soup = BeautifulSoup(kimnamhoon, 'lxml')    # BeautifulSoup 클래스의 매개변수를
#                                                 # 두 개를 사용하고 soup를 인스턴스화
# result = soup.find_all('ul')[2]
#
# print(result.li.div.text)



# from bs4 import BeautifulSoup
#
# with open('d:\data\ecologicalpyramid.html') as kimnamhoon:
#     soup = BeautifulSoup(kimnamhoon, 'lxml')    # BeautifulSoup 클래스의 매개변수를
#                                                 # 두 개를 사용하고 soup를 인스턴스화
# result = soup.find(text='fox')
#
# print(result)




# from bs4 import BeautifulSoup
#
# with open('d:\data\ecologicalpyramid.html') as kimnamhoon:
#     soup = BeautifulSoup(kimnamhoon, 'lxml')    # BeautifulSoup 클래스의 매개변수를
#                                                 # 두 개를 사용하고 soup를 인스턴스화
# result = soup.find_all('div', class_='number')[2]
#
# print(result.string)
#
# div_li_tags = soup.find_all(['div', 'li'])
# all_css_class = soup.find_all(class_=['producerlist', 'primaryconsumerslist'])




# import urllib.request
# from bs4 import BeautifulSoup
# import re
# import os
#
#
# def fetch_list_url():
#     list_url = "http://home.ebs.co.kr/ladybug/board/6/10059819/oneBoardList?hmpMnuId=106"
#     # print(list_url)
#     url = urllib.request.Request(list_url)
#     res = urllib.request.urlopen(url).read().decode("utf-8")
#     # print(res)  # 위의 두가지 작업을 거치면 위의 url의 html 문서를 res 변수에 담을 수 있게 된다
#     soup = BeautifulSoup(res, "html.parser")    # res 에 담긴 html 코드를 BeautifulSoup 모듈로 검색하기 위한 작업
#     # 위의 ebs 게시판 url 로 접속했을때 담긴 html 코드를
#     # soup 에 담겠다
#     # e_reg = re.compile("(레이디버그)")  # 완젼 이라는 텍스트를 검색하기 위해서 완젼이라는 한글을 컴파일
#
#     for link in soup.find_all('a'):
#         print(link.get('href'))
#
# fetch_list_url()




# import urllib.request
# from bs4 import BeautifulSoup
# import re
# import os
#
#
# def fetch_list_url():
#     list_url = "http://home.ebs.co.kr/ladybug/board/6/10059819/oneBoardList?hmpMnuId=106"
#     # print(list_url)
#     url = urllib.request.Request(list_url)
#     res = urllib.request.urlopen(url).read().decode("utf-8")
#     # print(res)  # 위의 두가지 작업을 거치면 위의 url의 html 문서를 res 변수에 담을 수 있게 된다
#     soup = BeautifulSoup(res, "html.parser")    # res 에 담긴 html 코드를 BeautifulSoup 모듈로 검색하기 위한 작업
#     # 위의 ebs 게시판 url 로 접속했을때 담긴 html 코드를
#     # soup 에 담겠다
#     # e_reg = re.compile("(레이디버그)")  # 완젼 이라는 텍스트를 검색하기 위해서 완젼이라는 한글을 컴파일
#
#     result = soup.find_all('p', class_='con')
#     result1 = soup.find_all('span', class_='date')
#     cnt = 0
#
#     for i in result:
#         print(result1[cnt].get_text(strip=True), i.get_text(strip=True))
#         cnt += 1
#
# fetch_list_url()




#
# import urllib.request
# from bs4 import BeautifulSoup
# import re
# import os
#
#
# def fetch_list_url():
#     list_url = "http://home.ebs.co.kr/ladybug/board/6/10059819/oneBoardList?hmpMnuId=106"
#     # print(list_url)
#     url = urllib.request.Request(list_url)
#     res = urllib.request.urlopen(url).read().decode("utf-8")
#     # print(res)  # 위의 두가지 작업을 거치면 위의 url의 html 문서를 res 변수에 담을 수 있게 된다
#     soup = BeautifulSoup(res, "html.parser")    # res 에 담긴 html 코드를 BeautifulSoup 모듈로 검색하기 위한 작업
#     # 위의 ebs 게시판 url 로 접속했을때 담긴 html 코드를
#     # soup 에 담겠다
#     # e_reg = re.compile("(레이디버그)")  # 완젼 이라는 텍스트를 검색하기 위해서 완젼이라는 한글을 컴파일
#
#     result = soup.find_all('p', class_='con')
#     result1 = soup.find_all('span', class_='date')
#     cnt = 0
#
#     for i in result:
#         print(result1[cnt].get_text(strip=True), i.get_text(strip=True))
#         cnt += 1c
#
# fetch_list_url()









import urllib.request
from bs4 import BeautifulSoup
import re
import os


def fetch_list_url(s):
    list_url = "http://home.ebs.co.kr/ladybug/board/6/10059819/oneBoardList?c.page="+str(s)+'&hmpMnuId=106&searchKeywordValue=0&bbsId=10059819&searchKeyword=&searchCondition=&searchConditionValue=0&'
    # print(list_url)
    url = urllib.request.Request(list_url)
    res = urllib.request.urlopen(url).read().decode("utf-8")
    # print(res)  # 위의 두가지 작업을 거치면 위의 url의 html 문서를 res 변수에 담을 수 있게 된다
    soup = BeautifulSoup(res, "html.parser")    # res 에 담긴 html 코드를 BeautifulSoup 모듈로 검색하기 위한 작업
    # 위의 ebs 게시판 url 로 접속했을때 담긴 html 코드를
    # soup 에 담겠다
    # e_reg = re.compile("(레이디버그)")  # 완젼 이라는 텍스트를 검색하기 위해서 완젼이라는 한글을 컴파일

    result = soup.find_all('p', class_='con')
    result1 = soup.find_all('span', class_='date')
    cnt = 0

    for i in result:
        print(result1[cnt].get_text(strip=True), i.get_text(strip=True))
        cnt += 1
        # print(i.get_text(strip=True))

for i in range(1,16):
    fetch_list_url(i)