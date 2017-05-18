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
    f = open(get_save_path(), 'w', encoding='utf-8')
    for i in range(1,13):
        for j in range(1,32):
            list_url = "https://www.wunderground.com/history/airport/RKSI/2016/"+str(i)+"/"+str(j)+"/DailyHistory.html"
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

            f.write(soup_mean_temp+', ')
            f.write(soup_max_temp+', ')
            f.write(soup_min_temp+', ')
            f.write(soup_wind+', ')
            f.write(soup_max_wind+', ')
            f.write(soup_vis+'\n')

    f.close()

fetch_list_url()