'''
- 페이스북 스크롤링
'''
#-*- coding: utf-8 -*-
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from datetime import datetime
import operator

class FacebookCrawler:
    # FILE_PATH = 'D:\\02.Python\\facebook_data\\'
    # CHROME_DRIVER_PATH = 'D:\\02.Python\\'
    FILE_PATH = 'D:\\facebook_data\\'
    CHROME_DRIVER_PATH = 'D:\\chromedriver\\'


    def __init__(self, searchKeyword, startMonth, endMonth, scroll_down_cnt):
        # 검색할 키워드, 시작날짜, 종료날짜, 마우스 스크롤 다운 횟수
        self.searchKeyword = searchKeyword
        self.startMonth = startMonth
        self.endMonth = endMonth
        self.scroll_down_cnt = scroll_down_cnt
        self.data = {}  # 게시날짜, 게시글 수집할 딕셔너리 변수
        self.url = 'https://www.facebook.com/search/str/' + searchKeyword + '/keywords_top?filters_rp_creation_time=%7B"start_month%22%3A"' + startMonth + '"%2C"end_month"%3A"' + endMonth + '"%7D'

        self.set_chrome_driver()    # 크롬 드라이브 위치를 지정하는 함수 실행
        self.play_crawling()        # 스크롤링 하는 함수 실행


    # chrome driver 생성 후 chrome 창 크기 설정하는 함수.
    def set_chrome_driver(self):
        self.driver = webdriver.Chrome(FacebookCrawler.CHROME_DRIVER_PATH + 'chromedriver.exe')
        # self.driver.set_window_size(1024, 768)    # 크롬 창 크기를 설정하는 코드


    # facebook 홈페이지로 이동 후 email, password 를 입력하고 submit 보내는 함수. (로그인)
    def facebook_login(self):
        self.driver.get("https://www.facebook.com/")    # 페이스북으로 이동
        self.driver.find_element_by_id("email").clear() # id 입력창 clear
        self.driver.find_element_by_id("email").send_keys("jm050106@naver.com")     # id 입력
        self.driver.find_element_by_id("pass").clear()                              # password 입력창 clear
        self.driver.find_element_by_id("pass").send_keys("sjm981690418!")           # password 입력
        self.driver.find_element_by_id("pass").submit()                             # 엔터
        time.sleep(5)   # 로그인시 시간이 걸리므로 잠깐 5초 sleep
        self.driver.get(self.url)   # 페이스북 검색 페이지로 이동

    # facebook page scroll down 하는 함수
    def page_scroll_down(self):
        for i in range(1, self.scroll_down_cnt):
            self.driver.find_element_by_xpath("//body").send_keys(Keys.END)
            time.sleep(3)


    # 크롤링 된 데이터를 파일로 저장하는 함수
    def data_to_file(self):
        with open(FacebookCrawler.FILE_PATH + self.searchKeyword + ".txt", "w", encoding="utf-8") as file:
            print('데이터를 저장하는 중입니다.')
            for key, value in sorted(self.data.items(), key=operator.itemgetter(0)):
                # data.items() 에 key 와 value 가 들어있고 그리고 0 번째 요소로 정렬하겠다.
                file.write(str(datetime.fromtimestamp(key)) + ' : ' + value + '\n')
            file.close()
            print('데이터 저장이 완료되었습니다.')


    # 크롤링 수행하는 메인 함수
    def play_crawling(self):
        try:
            self.facebook_login()   # 페이스북 로그인
            time.sleep(5)           # 잠깐 5초 sleep
            self.page_scroll_down() # 페이스북 스크롤 다운 실행
            html = self.driver.page_source  # 스크롤 다운한 현재까지의 html 문서를 담아서
            soup = BeautifulSoup(html, "html.parser")   # beautiful soup 로 검색할 수 있도록 설정

            #  . 이 클래스 # 이 id  붙이면 and 떨어뜨리면 or 조건
            # .fbUserContent._5pcr(클래스 and 클래스)
            # .fbUserContent _5pcr(클래수 or 클래스)
            for tag in soup.select('.fbUserContent._5pcr'):
                usertime = tag.find('abbr', class_='_5ptz')     # 게시한 날짜와 시간
                content = tag.find('div', class_='_5pbx userContent').find('p')     # 게시글
                if usertime is not None and content is not None:
                    self.data[int(usertime['data-utime'])] = content.get_text(strip=True)

            self.data_to_file()     # data_to_file() 함수를 실행해서 data 딕셔너리의 내용을 os의 파일로 생성
            self.driver.quit()
        except:
            print('정상 종료 되었습니다.')

crawler = FacebookCrawler('평창 올림픽', '2010-01', '2017-05', 20)
crawler.play_crawling()



'''
- 트위터 크롤링
'''
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import os


def get_save_path():    # 스크롤링할 텍스트 파일 생성할 위치지정 함수
    save_path = input("Enter the file name and file location :" )
    save_path = save_path.replace("\\", "/")
    if not os.path.isdir(os.path.split(save_path)[0]):
        os.mkdir(os.path.split(save_path)[0])
    return save_path


file = open(get_save_path(), 'w', encoding='utf-8')


binary = 'D:\chromedriver/chromedriver.exe'
browser = webdriver.Chrome(binary)  # 크롬 브라우저를 열겠다고 설정
browser.get("https://twitter.com/search-home")  # 트위터 검색 url로 접속
elem = browser.find_element_by_id("search-home-input")
#find_elements_by_class_name("")


elem.send_keys("삼성전자")  # 삼성전자로 검색창에 입력
elem.submit()              # 엔터


for i in range(1,15):    # 스크롤 다운 3번 수행
    browser.find_element_by_xpath("//body").send_keys(Keys.END)
    time.sleep(5)


time.sleep(5)
html = browser.page_source  # 내가 브라우져로 보고있는 소스를 볼려고하는것이다.
                            # 그런데 그냥 열면 사용자가 end 버튼틀 눌러서 컨트롤
                            # 한게 반영 안된것이 열린다.
soup = BeautifulSoup(html,"lxml")   # html 코드를 BeautifulSoup 에서 검색할 수 있도록 설정
#print(soup)
#print(len(soup))
tweet_tag = soup.find_all(class_="tweet-text")
#print(tweet_text)


for i in tweet_tag:
    tweet_text = i.get_text(strip=True)
    print(tweet_text)
    file.write(tweet_text)

file.close()

browser.quit()


'''
- 트위터 크롤링(클래스1)
'''
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import os


class TwitCrawler():

    def __init__(self,keyword,input_since,input_until,count):
        self.crawURL = 'https://twitter.com/search-advanced'
        self.FILE_PATH = 'D:\\twit\\twit.txt'
        self.keyword = str(keyword)
        self.input_since = str(input_since)
        self.input_until = str(input_until)
        self.count = int(count)

    def get_save_path(self):
        save_path = self.FILE_PATH.replace("\\", "/")
        if not os.path.isdir(os.path.split(save_path)[0]):
            os.mkdir(os.path.split(save_path)[0])
        return save_path


    def get_twit_data(self):

        binary = 'D:\chromedriver/chromedriver.exe'
        browser = webdriver.Chrome(binary)
        browser.get(self.crawURL)
        elem = browser.find_element_by_name("ands")
        since = browser.find_element_by_id("since")
        until = browser.find_element_by_id("until")
        #find_elements_by_class_name("")
        elem.send_keys(self.keyword)
        since.send_keys(self.input_since)
        until.send_keys(self.input_until)
        elem.submit()
        for i in range(1,self.count):
            browser.find_element_by_xpath("//body").send_keys(Keys.END)
            time.sleep(5)

        time.sleep(5)
        html = browser.page_source  # 내가 브라우져로 보고있는 소스를 볼려고하는것이다.
                                    # 그런데 그냥 열면 사용자가 end 버튼틀 눌러서 컨트롤
                                    # 한게 반영 안된것이 열린다.
        soup = BeautifulSoup(html,"lxml")
        #print(soup)
        #print(len(soup))
        self.tweet_tag = soup.find_all('div', class_="js-tweet-text-container")
        browser.quit()

    def write_twit_data(self):
        file = open(self.get_save_path(), 'w', encoding='utf-8')
        self.get_twit_data()
        for i in self.tweet_tag:
            tweet_text = i.get_text(strip=True)
            print(tweet_text)
            file.write(tweet_text)
        file.close()

twit = TwitCrawler('삼성전자','2015/01/01','2015/12/31',10)
twit.write_twit_data()



'''
- 트위터 크롤링(클래스2)
'''
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import operator

class TwitterCrawler:
    CHROME_DRIVER_PATH = 'D:\\chromedriver\\'
    FILE_PATH = 'D:\\twitter_data\\'
    def __init__(self,search_word,start_date,end_date,routine):
        self.search_word = search_word
        self.start_date = start_date
        self.end_date = end_date
        self.routine = routine
        self.data = {}
        self.url_form = 'https://twitter.com/search?l=&q='+search_word+'%20since%3A'+start_date+'%20until%3A'+end_date+'&src=typd&lang=ko'
        self.set_chrome_driver()
        self.play_crawling()
    def set_chrome_driver(self):
        self.driver = webdriver.Chrome(TwitterCrawler.CHROME_DRIVER_PATH + 'chromedriver.exe')

    def page_scroll_down(self):
        for i in range(0,self.routine):
            self.driver.find_element_by_xpath("//body").send_keys(Keys.END)
            time.sleep(5)

    def data_to_file(self):
        with open(TwitterCrawler.FILE_PATH + self.search_word + ".txt", "w", encoding="utf-8") as file: #PATH\키워드.txt를 쓰기가능한 유니코드파일로 열면서
            print('데이터를 저장하는 중입니다.')  #프린트하며
            for key, value in sorted(self.data.items(), key=operator.itemgetter(0)):   #data 딕셔너리에, 정렬하여 넣겠다.
                # data.items() 에 key 와 value 가 들어있고 그리고 0 번째 요소로 정령하겠다.
                file.write("==" * 30 + "\n")
                file.write(key + "\n")
                file.write(value + "\n")
                file.write("==" * 30 + "\n")


                file.write(value + '\n') #밸류값을 파일에 작성한다.
            file.close()  #파일종료
            print('데이터 저장이 완료되었습니다.')


    def play_crawling(self,):
        try:
            self.driver.get(self.url_form)
            self.page_scroll_down()
            html = self.driver.page_source
            soup = BeautifulSoup(html,"html.parser")
            content_find = soup.find_all("div",class_="content")  #len(18)
            for tag in content_find:
                usertime = tag.find('small', class_='time').find('a')['title'] #타이틀 자체가 값이라서 get_text 안함
                text = tag.find('p')
                # print(text)
                if usertime is not None and text is not None:
                    self.data[usertime] = text.get_text(strip=True)

            self.data_to_file()
            self.driver.quit()
            print(self.data)

        except:
            print('정상 종료 되었습니다.')

crawler = TwitterCrawler('문재인', '2015-02-01', '2017-05-01', 3)
crawler.play_crawling()



'''
- 인스타그램 이미지 다운 크롤링
'''
import urllib.request
from  bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

binary = 'D:\chromedriver/chromedriver.exe'
driver = webdriver.Chrome(binary)
driver.get("https://www.instagram.com/explore/")
driver.find_element_by_name("username").clear()
driver.find_element_by_name("username").send_keys("park_0902")
driver.find_element_by_name("password").clear()
driver.find_element_by_name("password").send_keys("sjm981690418!")
driver.find_element_by_name("password").submit()
# driver.find_element_by_xpath("//span[@id='react-root']/div/article/div/div/div/form/span/button").click()

# 검색어 입력
driver.get("https://www.instagram.com/explore/")
driver.find_element_by_css_selector("div._etslc").click()
elem = driver.find_element_by_css_selector("input._9x5sw._qy55y")
elem.clear()
elem.send_keys("포켓몬")
time.sleep(10)
elem.send_keys(Keys.ENTER)
time.sleep(10)

# 반복할 횟수
for i in range(1, 3):
    driver.find_element_by_xpath("//body").send_keys(Keys.END)
    time.sleep(5)
time.sleep(10)
html = driver.page_source
soup = BeautifulSoup(html, "lxml")

def fetch_list_url():
    params = []
    imgList = soup.find_all("img", class_="_icyx7")
    for im in imgList:
        params.append(im["src"])
    return params
    # print(params)

def fetch_detail_url():
    params = fetch_list_url()
    # print(params)
    a = 1
    for p in params:
        # 다운받을 폴더경로 입력
        urllib.request.urlretrieve(p, "D:/naverImages/" + str(a) + ".jpg")
        # download_web_images(p,'d:\');
        a = a + 1

fetch_detail_url()
driver.quit()


