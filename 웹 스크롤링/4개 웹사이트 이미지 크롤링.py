import urllib.request
from bs4 import BeautifulSoup
from selenium import webdriver   # 웹 애플리케이션의 테스트를 자동화하기 위한 프레임 워크
from selenium.webdriver.common.keys import Keys
import time    # sleep하기 위한 모듈




searchsite1 = {'naver':['https://search.naver.com/search.naver?where=image&amp;sm=stb_nmr&amp', 'nx_query', '_img', 'c:\\data\\naverImages\\'],
              'daum':['http://search.daum.net/search?nil_suggest=btn&w=img&DA=SBC&q=', 'q', 'thumb_img', 'c:\\data\\daumImages\\'],
               'google':['https://www.google.com/imghp?hl=ko','lst-ib','rg_ic rg_i','c:\\data\\googleimages\\'],
               'bing':['http://www.bing.com/?scope=images&FORM=Z9LH1','sb_form_q','mimg','c:\\data\\bingimages\\']}

searchsite2 = input('이미지를 검색할 웹브라우저 주소를 입력하세요(naver/daum/google/bing) ')

keyword = input('검색어를 입력하세요 : ')

# searchsite1['{}'.format(searchsite2)][2]

###########################url 받아오기###########################

# 크롬드라이버 경로설정(사전에 설치필요)
# 팬텀JS를 사용하면 백그라운드로 실행할 수 있다.
chrome = 'd:\chromedriver/chromedriver.exe'

browser = webdriver.Chrome(chrome)  # 웹브라우저 인스턴스화
browser.get(searchsite1['{}'.format(searchsite2)][0])
# 이미지를 검색할 웹사이트의 주소 입력(이미지만 검색하는 창을 추천한다.)

elem = browser.find_element_by_id(searchsite1['{}'.format(searchsite2)][1])  #naver.com같은 경우에는 "nx_query"


###########################검색어 입력#############################

elem.send_keys(keyword)  # 검색어 입력(검색어 입력창과 연결)
elem.submit()            # Enter키


###########################반복 횟수##############################

for i in range(1, 2):
    browser.find_element_by_xpath("//body").send_keys(Keys.END)
    #Enter키를 누르면 body를 활성화하겠다.(마우스로 클릭하는 개념)
    time.sleep(5)

time.sleep(7)                           # 네트워크의 상태를 고려하여 sleep
html = browser.page_source              # 크롬 브라우저에서 현재 불러온 소스를 가져온다.
soup = BeautifulSoup(html, "lxml")      # html코드를 검색할 수 있도록 설정


############################그림파일 저장##############################

def fetch_list_url():
    params = []
    imgList = soup.find_all("img", class_=searchsite1['{}'.format(searchsite2)][2])   # 네이버 이미지 url이 있는 태그의 _img클래스에 가서
    for img in imgList:
        try:
            params.append(img["src"])
        except KeyError:
            pass
    return params

def fetch_detail_url():
    params = fetch_list_url()
    for idx, img in enumerate(params, 1):
        urllib.request.urlretrieve(img, searchsite1['{}'.format(searchsite2)][3] + str(idx) + ".jpg")
        # 다운로드 받을 경로 입력

if __name__ == '__main__':
    # 메인실행 함수
    fetch_detail_url()

    #끝나면 브라우저 닫기
    browser.quit()










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