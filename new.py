import urllib.request           # 웹 브라우져에서 html 문서를 얻어오기위해 통신하기 위한 모듈
from bs4 import BeautifulSoup  # html 문서검색 모듈

def fetch_list_url():
    for i in range(1,10):
        list_url = "https://www.wunderground.com/history/airport/RKSI/2017/5/18/DailyHistory.html"
        url = urllib.request.Request(list_url) # url 요청에 따른 http 통신 헤더값을 얻어낸다
        res = urllib.request.urlopen(url).read().decode("utf-8")

        soup = BeautifulSoup(res, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
        soup_temp = soup.find_all('span', class_='wx-value')
        # print(soup_temp)
        soup_mean_temp = soup_temp[0].get_text()
        soup_max_temp = soup_temp[1].get_text()
        soup_min_temp = soup_temp[4].get_text()
        soup_wind = soup_temp[10].get_text()
        soup_max_wind = soup_temp[11].get_text()
        soup_vis = soup_temp[12].get_text()

        print(soup_mean_temp, soup_max_temp, soup_min_temp)
        print(soup_wind, soup_max_wind, soup_vis)
        # soup_td_span = soup_td.find('span', class_='nobr')
        # print(soup_td_span)
        # for i in range(p):
        #     soup_p = soup.find_all('p')[i]
        #     soup_p_a = soup_p.find('a')['href']
        #     print(soup_p_a)
        #
        # return soup_p_a

fetch_list_url()