# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# rawdate = pd.read_csv('d:\data7\\final_incheon_airport.csv', names=['Mean', 'Max', 'Min', 'dew','wind', 'maxwind', 'vis'])
# print(rawdate)
# print(rawdate.head(366))
# print(rawdate.plot())


# for i in range(1,10):
#     if i == 1:
#         i = '0'+str(i)
#     print(i)
#



from pylab import legend
from pylab import plot, show
from pylab import title, xlabel, ylabel
from pylab import axis
axis()


# title('average monthly humidity in Seoul, 2014')
# xlabel('Month')
# ylabel('Humidity')
# show()

month =range(1, 13)

seoul = [50.0, 52.0, 60.0, 60.0, 59.0, 73.0, 74.0, 77.0, 69.0, 63.0, 61.0, 56.0]
daejeon = [65.0, 63.0, 64.0, 61.0, 62.0, 77.0, 83.0, 87.0, 78.0, 76.0, 77.0, 75.0]
busan = [41.0, 62.0, 60.0, 59.0, 68.0, 80.0, 85.0, 86.0, 75.0, 68.0, 61.0, 50.0 ]
plot(month, seoul, month, daejeon, month, busan, marker="o")
axis(xmin=1, ymin=0)
title('average monthly humidity in Seoul, Daejeon, Busan (2014)')
xlabel('Month')
ylabel('Humidity')
legend(['Seoul', 'Daejeon', 'Busan'], loc='best')
show()











# import urllib.request           # 웹 브라우져에서 html 문서를 얻어오기위해 통신하기 위한 모듈
# from bs4 import BeautifulSoup  # html 문서검색 모듈
#
#
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
#     f = open(get_save_path(), 'w', encoding='utf-8')
#     # for i in range(1,13):
#     #     for j in range(1,32):
#     list_url = "https://www.wunderground.com/history/airport/RKSI/2016/1/2/DailyHistory.html"
#     url = urllib.request.Request(list_url) # url 요청에 따른 http 통신 헤더값을 얻어낸다
#     res = urllib.request.urlopen(url).read().decode("utf-8")
#
#     soup = BeautifulSoup(res, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
#     soup_temp = soup.find_all('span', class_='wx-value')
#             # print(soup_temp)
#     soup_mean_temp = soup_temp[0].get_text()
#     soup_max_temp = soup_temp[1].get_text()
#     soup_min_temp = soup_temp[4].get_text()
#     soup_wind = soup_temp[10].get_text()
#     soup_max_wind = soup_temp[11].get_text()
#     soup_vis = soup_temp[12].get_text()
#
#     f.write(soup_mean_temp+', ')
#     f.write(soup_max_temp+', ')
#     f.write(soup_min_temp+', ')
#     f.write(soup_wind+', ')
#     f.write(soup_max_wind+', ')
#     f.write(soup_vis+'\n')
#
#     f.close()
#
# fetch_list_url()




# import pandas as pd
# import numpy as np
#
# df = pd.read_csv("d:\data7\\final_incheon_airport.csv")
#
# df.colums = ['date','a','b','c','d','e','f']
#
# df['datetime'] = df['date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
# df['all_news_num'] = 1
# print(df.index)






