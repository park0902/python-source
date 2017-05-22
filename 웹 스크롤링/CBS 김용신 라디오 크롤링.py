import urllib.request  # 웹브라우저에서 html 문서를 얻어오기위해 통신하는 모듈
from  bs4 import BeautifulSoup  # html 문서 검색 모듈
import os
import re

def fetch_list_url():
    params = []
    for j in range(1, 30):
        for i in range(0, 15):
            list_url = "http://www.cbs.co.kr/radio/pgm/board.asp?page=" + str(j) + "&pn=list&skey=&sval=&bgrp=2&bcd=00350012&pcd=board&pgm=111&mcd=BOARD2"

            url = urllib.request.Request(list_url)
            res = urllib.request.urlopen(url).read()

            soup = BeautifulSoup(res, "html.parser")
            soup_a = soup.find_all('a', class_='bd_link')[i]['href']

            # 숫자와 ,가 포함되어져 있는 부분을 검색해서 다시 , 로 구분해서 분리
            soup_num = re.search("[0-9,',']{11}", soup_a).group().split(',')
            params.append(soup_num)

    return params


def fetch_detail_url():
    params = fetch_list_url()
    for i in params:
        list_url = 'http://www.cbs.co.kr/radio/pgm/board.asp?pn=read&skey=&sval=&anum='+str(i[1])+'&vnum='+str(i[0])+'&bgrp=2&page=2&bcd=00350012&pcd=board&pgm=111&mcd=BOARD2'
        url = urllib.request.Request(list_url)
        res = urllib.request.urlopen(url).read()

        soup = BeautifulSoup(res, "html.parser")
        title = soup.find('td', class_='bd_menu_content').get_text()
        content = soup.find('td', class_='bd_article').get_text()
        # print(content)

fetch_detail_url()







import urllib.request
from bs4 import BeautifulSoup
import re
import os
def get_save_path():
    save_path = input("Enter the file name and file location :" )
    save_path = save_path.replace("\\", "/")
    if not os.path.isdir(os.path.split(save_path)[0]):
        os.mkdir(os.path.split(save_path)[0])
    return save_path

def fetch_list_url():
    c = []   # 출력결과를 담을 리스트 변수
    for j in range(1,2):          # 페이지 수
        for i in range(15):
            list_url= "http://www.cbs.co.kr/radio/pgm/board.asp?page="+str(j)+"&pn=list&skey=&sval=&bgrp=2&bcd=00350012&pcd=board&pgm=111&mcd=BOARD2"
            # 페이지 번호만 바꿔서 for loop를 실행
            url = urllib.request.Request(list_url)
            res = urllib.request.urlopen(url).read()
            soup = BeautifulSoup(res, "html.parser")   # html 불러오기
            b = soup.find_all('a',class_="bd_link")[i]['href'] # 서울시와 같은 javaScript i번째 부분 가져오기 부분
            # xxxxxxjavascript:ViewArticle(5808,508146)가 아래에서 쓰인다.
            # 게시글을 클릭해서 들어가면 url 주소가 게시글 번호는 패턴이 있지만 508146은 딱히 패턴이 없습니다.
            #http://www.cbs.co.kr/radio/pgm/board.asp?pn=read&skey=&sval=&anum=508146vnum=5808&bgrp=2&page=1&bcd=00350012&pcd=board&pgm=111&mcd=BOARD2
            # print(b)
            # print('===================================================')
            d = re.search("[0-9,',']{11}", b).group().split(',')  # 숫자 중간에 ','가 있기 때문에 같이 찾아서 뽑아줍니다.
            c.append(d)
            # 최종 리스트 변수에 담기면 이런 모양입니다.['5823', '512304'], ['5822', '512277'], ['5821', '512158'] ...
    # print(c)
    return c   # fetch_list_url2()에서 사용하기 위해 return 합니다.





def fetch_list_url2():
    params = fetch_list_url()        # ['5823', '512304'], ['5822', '512277'], ['5821', '512158'] ... 를 params에 담아줍니다.
    f = open(get_save_path(), 'w', encoding="utf8")
    for i in params:   # params에 있는 요소 하나씩 for loop를 실행합니다.
        list_url ='http://www.cbs.co.kr/radio/pgm/board.asp?pn=read&skey=&sval=&anum='+i[1]+'vnum='+i[0]+'&bgrp=2&page=1&bcd=00350012&pcd=board&pgm=111&mcd=BOARD2'
        # page부분은 1이나2나 상관없이 긁어오는데 맞추실 분들은 page번호도 맞추시면 됩니다.
        # params의 ['5823', '512304']를 url에 입력해줍니다.
        url = urllib.request.Request(list_url)
        res = urllib.request.urlopen(url).read()
        soup = BeautifulSoup(res, "html.parser")
        title = soup.find('td', class_="bd_menu_content").get_text()     # 게시글의 제목을 글자만 가져와서 title에 담습니다.
        article = soup.find('td' , class_="bd_article").get_text()       #  게시글 안의 내용을 글자만 가져와서 article에 담습니다.
        f.write(title + "\n" + article + "\n")   # 제목을 입력하고 엔터 게시 내용을 입력하고 엔터를 입력해줍니다.
        f.write("=======================================================================================================================\n\n\n")
        # 다음글과 구분을 위해 입력해줍니다.
    f.close()    # 파일 닫기

fetch_list_url2()
