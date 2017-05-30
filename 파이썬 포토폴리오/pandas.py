'''
- 2016년 최고기온 그래프
'''
import matplotlib.pyplot as plt
import pandas as pd

fig = plt.figure(); ax = fig.add_subplot(1,1,1)

df = pd.read_csv("d:\data\\final_incheon_airport1.csv")
mean = list(df['max'])
month =range(1, 367)

plt.plot(month, mean, label='Max Temperauture', c="y")

ax.set_xticks([0,31,60,91,121,151,181,212,243,273,304,334.366])
ax.set_xticklabels(['January','February','March','April','May','Jun','July','August','September',
                             'October','November','December'], rotation=30)

ax.set_title('RKSI Max Temperature (2016)')
plt.xlabel('Months')
plt.ylabel('Temperauture(°C)')
ax.legend(loc='best')

plt.show()




'''
- 2월(겨울)과 8월(여름)의 풍속 그래프 
'''
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("d:\data\\final_incheon_airport1.csv")
month = range(1, 30)
month1 = range(1, 32)

wind_1 = list(df['wind'][df['month']==2])
wind_8 = list(df['wind'][df['month']==8])

plt.xlim(1, 32)
plt.ylim(0, 40)

plt.subplot(2,1,1)
plt.plot(month, wind_1, 'yo-')
plt.title(' January & August Wind Speed (2016)')
plt.xlabel('January')
plt.ylabel('Wind speed(km/h)')

plt.subplot(2,1,2)
plt.plot(month1, wind_8, 'r.-')
plt.xlabel('August')
plt.ylabel('Wind speed(km/h)')

plt.show()




'''
- 월별 평균 기온 그래프 그리기
'''
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("d:\data\\final_incheon_airport1.csv")

month1 = range(1, 30)
month2 = range(1, 31)
month3 = range(1, 32)

re1 = list(df['mean'][df['month']==1])
re2 = list(df['mean'][df['month']==2])
re3 = list(df['mean'][df['month']==3])
re4 = list(df['mean'][df['month']==4])
re5 = list(df['mean'][df['month']==5])
re6 = list(df['mean'][df['month']==6])
re7 = list(df['mean'][df['month']==7])
re8 = list(df['mean'][df['month']==8])
re9 = list(df['mean'][df['month']==9])
re10 = list(df['mean'][df['month']==10])
re11 = list(df['mean'][df['month']==11])
re12 = list(df['mean'][df['month']==12])

plt.xlim(1, 32)
plt.ylim(-20, 40)

plt.subplot(341); plt.plot(month3, re1); plt.title('January Avg Temp'); plt.xlabel('Days'); plt.ylabel('Temperature(°C)')
plt.subplot(342); plt.plot(month1, re2); plt.title('February Avg Temp'); plt.xlabel('Days'); plt.ylabel('Temperature(°C)')
plt.subplot(343); plt.plot(month3, re3); plt.title('March Avg Temp'); plt.xlabel('Days'); plt.ylabel('Temperature(°C)')
plt.subplot(344); plt.plot(month2, re4); plt.title('April Avg Temp'); plt.xlabel('Days'); plt.ylabel('Temperature(°C)')
plt.subplot(345); plt.plot(month3, re5); plt.title('May Avg Temp'); plt.xlabel('Days'); plt.ylabel('Temperature(°C)')
plt.subplot(346); plt.plot(month2, re6); plt.title('Jun Avg temp'); plt.xlabel('Days'); plt.ylabel('Temperature(°C)')
plt.subplot(347); plt.plot(month3, re7); plt.title('July Avg Temp'); plt.xlabel('Days'); plt.ylabel('Temperature(°C)')
plt.subplot(348); plt.plot(month3, re8); plt.title('August Avg Temp'); plt.xlabel('Days'); plt.ylabel('Temperature(°C)')
plt.subplot(349); plt.plot(month2, re9); plt.title('September Avg Temp'); plt.xlabel('Days'); plt.ylabel('Temperature(°C)')
plt.subplot(3,4,10); plt.plot(month3, re10); plt.title('October Avg Temp'); plt.xlabel('Days'); plt.ylabel('Temperature(°C)')
plt.subplot(3,4,11); plt.plot(month2, re11); plt.title('November Avg Temp'); plt.xlabel('Days'); plt.ylabel('Temperature(°C)')
plt.subplot(3,4,12); plt.plot(month3, re12); plt.title('December Avg Temp'); plt.xlabel('Days'); plt.ylabel('Temperature(°C)')
plt.tight_layout()

plt.show()




'''
- 여름(6,7,8월) 최고 기온 그래프
'''
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("d:\data\\final_incheon_airport1.csv")

month1 = range(1, 30)
month2 = range(1, 31)
month3 = range(1, 32)

re6 = list(df['max'][df['month']==6])
re7 = list(df['max'][df['month']==7])
re8 = list(df['max'][df['month']==8])

plt.plot(month2, re6, label='Jun', c="b", lw=1, ls=":", marker="D")
plt.hold(True)
plt.plot(month3, re7, label='July', c="g", lw=1, ls=":", marker="s")
plt.hold(True)
plt.plot(month3, re8, label='August', c="y", lw=1, ls=":", marker="o")
plt.hold(True)

plt.title('Summer Max Temperature')
plt.xlabel('Days')
plt.ylabel('Temperature(°C)')
plt.legend(loc=3)

plt.show()



'''
- 3월 최저 온도 막대그래프(수직)
'''
import pandas as pd

df = pd.read_csv("d:\data\\final_incheon_airport1.csv")
vis = list(df['min'][df['month'] == 3])

month = range(1, 32)
df = pd.DataFrame(vis, index=month, columns=pd.Index(['March Min Temperature']))

df.plot(kind='bar', color='b', alpha=0.5)



'''
- 8월 최고풍속 막대그래프(수평)
'''
import pandas as pd

df = pd.read_csv("d:\data\\final_incheon_airport1.csv")
vis = list(df['maxwind'][df['month']==8])

month = range(1, 32)
df = pd.DataFrame(vis, index=month, columns=pd.Index(['March Max Temperature']))

df.plot(kind='barh', color='y')





'''
- 온도와 이슬점 관계
'''
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("d:\data\\final_incheon_airport1.csv")
fig = plt.figure(); ax = fig.add_subplot(1,1,1)

year = range(1, 367)

mean = list(df['mean'])
dew = list(df['dew'])
# re = mean-dew
ax.set_xticks([0,31,60,91,121,151,181,212,243,273,304,334.366])
ax.set_xticklabels(['January','February','March','April','May','Jun','July','August','September',
                             'October','November','December'], rotation=30)

plt.plot(year, mean, label='Mean Temperature', )
plt.hold(True)
plt.plot(year, dew, label='Dew point')
plt.hold(True)

ax.set_title('RKSI Max Temperature (2016)')
plt.xlabel('Months')
plt.ylabel('Temperauture(°C)')
ax.legend(loc=2)

plt.show()






'''
- 이슬점-온도 & 가시도 관계
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("d:\data\\final_incheon_airport1.csv")

year = range(1, 367)

mean = np.array(df['mean'])
dew = np.array(df['dew'])
re = list(mean-dew)
vis = list(df['vis'])

# print(mean-dew)
# print(re)
plt.plot(year, re, ls=':',marker='.',label='mean-dew', )
plt.hold(True)
plt.plot(year, vis, label='vis')
plt.hold(False)

plt.title('2016 RKSI MeanTemperature & Dew point')
plt.yticks([0, 10, 20])
plt.xlabel('Days')
plt.ylabel('Temperature')

plt.legend(loc=2)
# plt.legend(loc=1)

plt.show()




'''
- 가시도 & (온도-이슬점) 관계 산포도
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("d:\data\\final_incheon_airport1.csv")

mean = np.array(df['mean'])
dew = np.array(df['dew'])
re = list(mean-dew)
vis = list(df['vis'])

plt.scatter(vis, re)
plt.title('2016 RKSI Dew point & (Mean Temperature - Dew Point)')
plt.ylabel('(Mean Temperature - Dew Point)')
plt.xlabel('Visibility')



'''
- 가시도 & (온도-이슬점) 관계 산포도 및 선형 회귀
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("d:\data\\final_incheon_airport1.csv")
mean = np.array(df['mean'])
dew = np.array(df['dew'])
re = mean-dew
vis = np.array(df['vis'])

A = np.vstack([re, np.ones(len(re))]).T

a,b = np.linalg.lstsq(A, vis)[0]

plt.plot(vis, re, 'o', label='data', markersize=8)
plt.hold(True)
plt.plot(a*re + b, re, 'r', label='Fitted Line')
plt.hold(False)
plt.legend()
plt.show()

plt.title('2016 RKSI Dew point & (Mean Temperature - Dew Point)')
plt.ylabel('(Mean Temperature - Dew Point)')
plt.xlabel('Visibility')

print('가시도 = ','(온도-이슬점) * ',a ,' + ', b)











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












# pd.scatter_matrix(re, diagonal='kde', color='k', alpha=0.3)



year = range(1, 367)

mean = np.array(df['mean'])
dew = np.array(df['dew'])
re = list(mean-dew)
vis = list(df['vis'])

# print(mean-dew)
# print(re)
# plt.plot(year, re, label='mean-dew', )
# plt.hold(True)
plt.plot(year, vis, label='vis')
plt.hold(True)

plt.title('2016 RKSI MeanTemperature & Dew point')
plt.xlabel('Days')
plt.ylabel('Temperature')

# plt.legend(loc=2)
plt.legend(loc=1)

plt.show()





















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




import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("d:\data\\final_incheon_airport1.csv")
month = range(1, 32)
month1 = range(1, 33)

re = list(df['max'][df['month']==1])
re1 = list(df['max'][df['month']==8])

plt.xlim(1, 32)
plt.ylim(-20, 40)

plt.subplot(1,1,1)
plt.plot(month, re, 'yo-')
plt.title('average monthly humidity in Seoul, Daejeon, Busan (2014)')
plt.ylabel('Humidity')

# ax2 = plt.subplot(2,1,2)
# plt.plot(month, re1, 'r.-')
# plt.xlabel('Month')
# plt.ylabel('Humidity')

# df = pd.DataFrame(vis, index=month, columns=pd.Index(['Visibility']))
#
# df.plot(kind='bar', stacked=True, alpha=0.5)


plt.show()

























import numpy as np
import matplotlib.pyplot as plt

##임산부의 에스트리올 수치(x)가 출생시 체중(y)에 미치는 영향?
x = np.array([7,9,9,12,14,16,16,14,16,16,17,19,21,24,15,16,17,25,27,15,15,15,16,19,18,17,18,20,22,25,24])
y = np.array([25,25,25,27,27,27,24,30,30,31,30,31,30,28,32,32,32,32,34,34,34,35,35,34,35,36,37,38,40,39,43])
A = np.vstack([x, np.ones(len(x))]).T
print(x)
#a=알파, b=베타
a, b = np.linalg.lstsq(A, y)[0] #Return the least-squares solution to a linear matrix equation.

#산점도
plt.plot(x, y, 'o', label='data', markersize=8)
plt.hold(True)
plt.plot(x, a*x + b, 'r', label='Fitted line')
plt.hold(False)
plt.legend()
plt.show()
#회귀식
print('출생시 체중 = ',a,' * 에스트리올 + ',b)
# 출생시 체중 =  0.60819047619  * 에스트리올 +  21.5234285714