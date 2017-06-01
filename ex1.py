csv_file = input('파일경로와 파일명 입력: ')
# question = int(input('보고싶은 시각화 선택(1. 카테고리 그래프 2. 월별 카테고리 그래프 3. 여름 카테고리별 그래프 4. 카테고리별 바 그래프(수평))'))
# category = input('카테고리 입력(mean, max, min, dew, wind, maxwind, vis)')

# d:\data\\final_incheon_airport1.csv

'''
- 2016년 최고기온 그래프
'''
import matplotlib.pyplot as plt
import pandas as pd

fig = plt.figure(); ax = fig.add_subplot(1,1,1)

df = pd.read_csv(csv_file)
max = list(df['max'])
month =range(1, 367)

plt.plot(month, max, label='Max Temperauture', c="y")

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

df = pd.read_csv(csv_file)
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

df = pd.read_csv(csv_file)

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

df = pd.read_csv(csv_file)

month1 = range(1, 30)
month2 = range(1, 31)
month3 = range(1, 32)

re6 = list(df['max'][df['month']==6])
re7 = list(df['max'][df['month']==7])
re8 = list(df['max'][df['month']==8])

plt.plot(month2, re6, label='Jun', c="b", lw=1, ls=":", marker="D")
# plt.hold(True)
plt.plot(month3, re7, label='July', c="g", lw=1, ls=":", marker="s")
# plt.hold(True)
plt.plot(month3, re8, label='August', c="y", lw=1, ls=":", marker="o")
# plt.hold(True)

plt.title('Summer Max Temperature')
plt.xlabel('Days')
plt.ylabel('Temperature(°C)')
plt.legend(loc=3)

plt.show()



import matplotlib as mpl

a = set(sorted([f.name for f in mpl.font_manager.fontManager.ttflist]))

for i in a:
    print(i)



import matplotlib.pyplot as plt
import numpy as np
mpl.rc('font', family='Malgun Gothic')
mpl.rc('axes', unicode_minus=False)


x = np.linspace(0.0, 5.0, 100)
y = np.cos(2*np.pi*x) * np.exp(-x)
plt.title(u'한글 제목')
plt.plot(x, y, label=u"코사인")
t = 2 * np.pi / 3
plt.scatter(t, np.cos(t), 50, color='blue')
plt.xlabel(u"엑스축 라벨")
plt.ylabel(u"와이축 라벨")
plt.annotate(u"여기가 0.5!", xy=(t, np.cos(t)), xycoords='data', xytext=(-90, -50),
             textcoords='offset points', fontsize=16, arrowprops=dict(arrowstyle="->"))
plt.show()










import matplotlib.pyplot as plt
import pandas as pd


mpl.rc('font', family='Malgun Gothic')
mpl.rc('axes', unicode_minus=False)

df = pd.read_csv(csv_file)
month = range(1, 30)
month1 = range(1, 32)

wind_1 = list(df['wind'][df['month']==2])
wind_8 = list(df['wind'][df['month']==8])

plt.xlim(1, 32)
plt.ylim(0, 40)

plt.subplot(2,1,1)
plt.plot(month, wind_1, 'yo-')
plt.title(u"가나다라마바사")
plt.xlabel(u'가나다라 January')
plt.ylabel('Wind speed(km/h)')

plt.subplot(2,1,2)
plt.plot(month1, wind_8, 'r.-')
plt.xlabel('August')
plt.ylabel('Wind speed(km/h)')

plt.show()