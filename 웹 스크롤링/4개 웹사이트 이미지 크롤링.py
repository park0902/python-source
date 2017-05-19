'''
- 네이버 이미지 다운로드
'''

# import urllib.request
# from  bs4 import BeautifulSoup
# from selenium import webdriver      # 웹 어플리케이션의 테스트를 자동화하기 위한 프레임 워크
# from selenium.webdriver.common.keys import Keys
# import time         # 중간중간 sleep을 걸어야해서 time 모듈을 import 한다
#
# binary = 'D:\chromedriver/chromedriver.exe'
# # 웹 브라우저를 크롬을 사용할거라서 크롬 드라이버를 다운받아서 위의 위치에 둔다
# # 팬텀 js 로 하면 백 그라운드로 실행할 수 있다
#
# browser = webdriver.Chrome(binary)  # 브라우저를 인스턴스화
# browser.get("https://search.naver.com/search.naver?where=image&amp;sm=stb_nmr&amp;")
# # 네이버의 이미지 검색 url 을 받아온다
#
# elem = browser.find_element_by_id("nx_query")
# # 네이버의 이미지 검색에 해당하는 input 창의 id 가 nx_query여서 검색창의 해당하는 hteml 코드를
# # 찾아서 eleml요소 사용하겠금 설정
#
# # find_elements_by_class_name("") # 클래스 이름을 찾을때 방법
#
# # 검색어 입력
# elem.send_keys("어밴져스")  # elem 이 input 창과 연결이 되어서 스스로 아이언맨을 검색창에 쓴다
# elem.submit()              # 웹에서의 submit 은 엔터의 역할
#
# # 반복할 횟수
# for i in range(1,2):   # 2번만 누르게 한다
#     browser.find_element_by_xpath("//body").send_keys(Keys.END)
#     # 브라우저 아무데서나 end 키를 누른다고 해서 페이지가 아래로 내려가지 않아서 body 활성화시킨후 end 키를 누른다
#
#     time.sleep(5)   # end 로 내려오는데 시간이 걸려서 5초의 sleep
#
# time.sleep(5)       # 네트워크 안정성(느릴까봐) 5초의 sleep
#
# html = browser.page_source  # 크롬 브라우저에서 현재 불러온 소스를 가져온다
#
# soup = BeautifulSoup(html, "lxml")  # BeautifulSoup 를 사용해서 html 코드를 검색할 수 있도록 설정
#
# # print(soup)
# # print(len(soup))
#
#
# def fetch_list_url():
#     params = []
#     imgList = soup.find_all("img", class_="_img")   # 네이버 이미지 url 이 있는 img 태그의 _img 클래스로 가서
#     for im in imgList:
#         params.append(im["src"])                    # params 리스트 변수에 image_url 을 담는다
#     return params
#
#
# def fetch_detail_url():
#     params = fetch_list_url()
#     print(params)
#     a = 1
#     for p in params:
#         print(p)
#         # 다운받을 폴더경로 입력
#         urllib.request.urlretrieve(p, "d:/naverImages/" + str(a) + ".jpg")
#         a += 1
#
# fetch_detail_url()
#
# browser.quit()





'''
- 다음 이미지 다운로드
'''
# import urllib.request
# from  bs4 import BeautifulSoup
# from selenium import webdriver      # 웹 어플리케이션의 테스트를 자동화하기 위한 프레임 워크
# from selenium.webdriver.common.keys import Keys
# import time         # 중간중간 sleep을 걸어야해서 time 모듈을 import 한다
#
# binary = 'D:\chromedriver/chromedriver.exe'
# # 웹 브라우저를 크롬을 사용할거라서 크롬 드라이버를 다운받아서 위의 위치에 둔다
# # 팬텀 js 로 하면 백 그라운드로 실행할 수 있다
#
# browser = webdriver.Chrome(binary)  # 브라우저를 인스턴스화
# browser.get("http://search.daum.net/search?nil_suggest=btn&w=img&DA=SBC&q=")
# # 네이버의 이미지 검색 url 을 받아온다
#
# elem = browser.find_element_by_id("q")
# # 네이버의 이미지 검색에 해당하는 input 창의 id 가 nx_query여서 검색창의 해당하는 hteml 코드를
# # 찾아서 eleml요소 사용하겠금 설정
#
# # find_elements_by_class_name("") # 클래스 이름을 찾을때 방법
#
# # 검색어 입력
# elem.send_keys("어밴져스")  # elem 이 input 창과 연결이 되어서 스스로 아이언맨을 검색창에 쓴다
# elem.submit()              # 웹에서의 submit 은 엔터의 역할
#
# # 반복할 횟수
# for i in range(1,2):   # 2번만 누르게 한다
#     browser.find_element_by_xpath("//body").send_keys(Keys.END)
#     # 브라우저 아무데서나 end 키를 누른다고 해서 페이지가 아래로 내려가지 않아서 body 활성화시킨후 end 키를 누른다
#
#     time.sleep(5)   # end 로 내려오는데 시간이 걸려서 5초의 sleep
#
# time.sleep(5)       # 네트워크 안정성(느릴까봐) 5초의 sleep
#
# html = browser.page_source  # 크롬 브라우저에서 현재 불러온 소스를 가져온다
#
# soup = BeautifulSoup(html, "lxml")  # BeautifulSoup 를 사용해서 html 코드를 검색할 수 있도록 설정
#
# # print(soup)
# # print(len(soup))
#
#
# def fetch_list_url():
#     params = []
#     imgList = soup.find_all("img", class_="thumb_img")   # 다음 이미지 url 이 있는 img 태그의 _img 클래스로 가서
#     for im in imgList:
#         params.append(im["src"])                    # params 리스트 변수에 image_url 을 담는다
#     return params
#
#
# def fetch_detail_url():
#     params = fetch_list_url()
#     print(params)
#     a = 1
#     for p in params:
#         print(p)
#         # 다운받을 폴더경로 입력
#         urllib.request.urlretrieve(p, "d:/daumImage/" + str(a) + ".jpg")
#         a += 1
#
# fetch_detail_url()
#
# browser.quit()



'''
- 구글 이미지 다운로드
'''
# import urllib.request
# from  bs4 import BeautifulSoup
# from selenium import webdriver  # 웹 애플리케이션의 테스트를 자동화하기 위한 프레임 워크
# from selenium.webdriver.common.keys import Keys
# import time                     # 중간중간 sleep 을 걸어야 해서 time 모듈 import
#
# ########################### url 받아오기 ###########################
#
# # 웹브라우져로 크롬을 사용할거라서 크롬 드라이버를 다운받아 위의 위치에 둔다
# # 팬텀 js로 하면 백그라운드로 실행할 수 있음
# binary = 'D:\chromedriver/chromedriver.exe'
#
# # 브라우져를 인스턴스화
# browser = webdriver.Chrome(binary)
#
# # 네이버의 이미지 검색 url 받아옴(아무것도 안 쳤을때의 url)
# browser.get("https://www.google.co.kr/imghp?hl=ko&tab=wi&ei=l1AdWbegOcra8QXvtr-4Cw&ved=0EKouCBUoAQ")
#
# # 네이버의 이미지 검색에 해당하는 input 창의 id 가 'nx_query' 임(검색창에 해당하는 html코드를 찾아서 elem 사용하도록 설정)
# # input창 찾는 방법은 원노트에 있음
# # find_elements_by_class_name("") --> 클래스 이름으로 찾을때는 이렇게
#
# elem = browser.find_element_by_id("lst-ib")
#
# ########################### 검색어 입력 ###########################
#
# # elem 이 input 창과 연결되어 스스로 아이언맨을 검색
# elem.send_keys("꼬부기")
#
# # 웹에서의 submit 은 엔터의 역할을 함
# elem.submit()
# ########################### 반복할 횟수 ###########################
# # 스크롤을 내리려면 브라우져 이미지 검색결과 부분(바디부분)에 마우스 클릭 한번 하고 End키를 눌러야함
# for i in range(1, 2):
#     browser.find_element_by_xpath("//body").send_keys(Keys.END)
#     time.sleep(10)                  # END 키 누르고 내려가는데 시간이 걸려서 sleep 해줌
#
# time.sleep(10)                      # 네트워크 느릴까봐 안정성 위해 sleep 해줌
# html = browser.page_source         # 크롬브라우져에서 현재 불러온 소스 가져옴
# soup = BeautifulSoup(html, "lxml") # html 코드를 검색할 수 있도록 설정
#
# ########################### 그림파일 저장 ###########################
#
# def fetch_list_url():
#     params = []
#     imgList = soup.find_all("img", class_="rg_ic rg_i")  # 네이버 이미지 url 이 있는 img 태그의 _img 클래스에 가서
#     for im in imgList:
#         try :
#             params.append(im["src"])                   # params 리스트에 image url 을 담음
#         except KeyError:
#             params.append(im["data-src"])
#     return params
#
# def fetch_detail_url():
#     params = fetch_list_url()
#     for idx,p in enumerate(params,1):
#         # 다운받을 폴더경로 입력
#         urllib.request.urlretrieve(p, "D:\googleImages/" + str(idx) + ".jpg")
# if __name__ == '__main__':
#     # 메인 실행 함수
#     fetch_detail_url()
#     # 끝나면 브라우져 닫기
#     browser.quit()



'''
- 빙 이미지 다운로드
'''
# import urllib.request
# from  bs4 import BeautifulSoup
# from selenium import webdriver  # 웹 애플리케이션의 테스트를 자동화하기 위한 프레임 워크
# from selenium.webdriver.common.keys import Keys
# import time                     # 중간중간 sleep 을 걸어야 해서 time 모듈 import
# binary = 'D:\chromedriver/chromedriver.exe'
# browser = webdriver.Chrome(binary)
# browser.get("https://www.bing.com/?scope=images&FORM=Z9LH1")  # 빙 이미지 검색 url
# elem = browser.find_element_by_id("sb_form_q")
#
# # 검색어 입력
# elem.send_keys("아이언맨")
# elem.submit()
#
# # 반복할 횟수
# for i in range(1, 2):
#     browser.find_element_by_xpath("//body").send_keys(Keys.END)
#
#     time.sleep(5)
# time.sleep(5)
# html = browser.page_source
# soup = BeautifulSoup(html, "lxml")
#
#
# def fetch_list_url():
#     params = []
#     imgList = soup.find_all("img", class_="mimg")
#     for im in imgList:
#         params.append(im["src"])
#     return params
#
#
# def fetch_detail_url():
#     params = fetch_list_url()
#     a = 1
#     for p in params:
#         urllib.request.urlretrieve(p, "d:/bingimages/" + str(a) + ".jpg")
#         a = a + 1
#
#
# fetch_detail_url()
#
# browser.quit()




'''
- 4개의 웹 사이트를 합친 이미지 다운로드 크롤링
'''
# import urllib.request
# from bs4 import BeautifulSoup
# from selenium import webdriver   # 웹 애플리케이션의 테스트를 자동화하기 위한 프레임 워크
# from selenium.webdriver.common.keys import Keys
# import time    # sleep하기 위한 모듈
#
#
#
#
# searchsite1 = {'naver':['https://search.naver.com/search.naver?where=image&amp;sm=stb_nmr&amp', 'nx_query', '_img', 'c:\\data\\naverImages\\'],
#               'daum':['http://search.daum.net/search?nil_suggest=btn&w=img&DA=SBC&q=', 'q', 'thumb_img', 'c:\\data\\daumImages\\'],
#                'google':['https://www.google.com/imghp?hl=ko','lst-ib','rg_ic rg_i','c:\\data\\googleimages\\'],
#                'bing':['http://www.bing.com/?scope=images&FORM=Z9LH1','sb_form_q','mimg','c:\\data\\bingimages\\']}
#
# searchsite2 = input('이미지를 검색할 웹브라우저 주소를 입력하세요(naver/daum/google/bing) ')
#
# keyword = input('검색어를 입력하세요 : ')
#
# # searchsite1['{}'.format(searchsite2)][2]
#
# ###########################url 받아오기###########################
#
# # 크롬드라이버 경로설정(사전에 설치필요)
# # 팬텀JS를 사용하면 백그라운드로 실행할 수 있다.
# chrome = 'd:\chromedriver/chromedriver.exe'
#
# browser = webdriver.Chrome(chrome)  # 웹브라우저 인스턴스화
# browser.get(searchsite1['{}'.format(searchsite2)][0])
# # 이미지를 검색할 웹사이트의 주소 입력(이미지만 검색하는 창을 추천한다.)
#
# elem = browser.find_element_by_id(searchsite1['{}'.format(searchsite2)][1])  #naver.com같은 경우에는 "nx_query"
#
#
# ###########################검색어 입력#############################
#
# elem.send_keys(keyword)  # 검색어 입력(검색어 입력창과 연결)
# elem.submit()            # Enter키
#
#
# ###########################반복 횟수##############################
#
# for i in range(1, 2):
#     browser.find_element_by_xpath("//body").send_keys(Keys.END)
#     #Enter키를 누르면 body를 활성화하겠다.(마우스로 클릭하는 개념)
#     time.sleep(5)
#
# time.sleep(7)                           # 네트워크의 상태를 고려하여 sleep
# html = browser.page_source              # 크롬 브라우저에서 현재 불러온 소스를 가져온다.
# soup = BeautifulSoup(html, "lxml")      # html코드를 검색할 수 있도록 설정
#
#
# ############################그림파일 저장##############################
#
# def fetch_list_url():
#     params = []
#     imgList = soup.find_all("img", class_=searchsite1['{}'.format(searchsite2)][2])   # 네이버 이미지 url이 있는 태그의 _img클래스에 가서
#     for img in imgList:
#         try:
#             params.append(img["src"])
#         except KeyError:
#             pass
#     return params
#
# def fetch_detail_url():
#     params = fetch_list_url()
#     for idx, img in enumerate(params, 1):
#         urllib.request.urlretrieve(img, searchsite1['{}'.format(searchsite2)][3] + str(idx) + ".jpg")
#         # 다운로드 받을 경로 입력
#
# if __name__ == '__main__':
#     # 메인실행 함수
#     fetch_detail_url()
#
#     #끝나면 브라우저 닫기
#     browser.quit()




'''
- 네이버 자동로그인 크롤링
'''
# import urllib.request
# from  bs4 import BeautifulSoup
# import time
# from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
#
#
#
# binary = 'd:\chromedriver/chromedriver.exe'
# browser = webdriver.Chrome(binary)
# browser.get("https://nid.naver.com/nidlogin.login?url=http%3A%2F%2Fmail.naver.com%2F")
#
#
# id = browser.find_element_by_id("id")
# pw = browser.find_element_by_id("pw")
# time.sleep(3)
# id.send_keys("jm050106")
# id.submit()
# pw.send_keys("sjm981690418!")
# pw.submit()







import urllib.request
from  bs4 import BeautifulSoup
from selenium import webdriver  # 웹 애플리케이션의 테스트를 자동화하기 위한 프레임 워크
from selenium.webdriver.common.keys import Keys
import time                     # 중간중간 sleep 을 걸어야 해서 time 모듈 import
binary = 'D:\chromedriver/chromedriver.exe'
browser = webdriver.Chrome(binary)
browser.get("https://www.bing.com/?scope=images&FORM=Z9LH1")  # 빙 이미지 검색 url
elem = browser.find_element_by_id("sb_form_q")

# 검색어 입력
elem.send_keys("인공지능")
elem.submit()

# 반복할 횟수
for i in range(1, 2):
    browser.find_element_by_xpath("//body").send_keys(Keys.END)

    time.sleep(5)
time.sleep(5)
html = browser.page_source
soup = BeautifulSoup(html, "lxml")


def fetch_list_url():
    params = []
    imgList = soup.find_all("img", class_="mimg")
    for im in imgList:
        params.append(im["src"])
    return params


def fetch_detail_url():
    params = fetch_list_url()
    a = 1
    for p in params:
        urllib.request.urlretrieve(p, "d:/bingimages/" + str(a) + ".jpg")
        a = a + 1


fetch_detail_url()

browser.quit()