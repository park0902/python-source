'''
- 2016년 최고기온 그래프
'''
import matplotlib.pyplot as plt
import pandas as pd

fig = plt.figure(); ax = fig.add_subplot(1,1,1)

df = pd.read_csv("d:\data\\final_incheon_airport1.csv")
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
- 8월 최고풍속 막대그래프(수평)
'''
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rc
from matplotlib import style
import pandas as pd

df = pd.read_csv("d:\data\\final_incheon_airport1.csv")

maxwind = list(df['maxwind'][df['month'] == 8])

month = range(1, 32)

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
style.use('ggplot')

industry = month
fluctuations = maxwind
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)

ypos = np.arange(31)
rects = plt.barh(ypos, fluctuations, align='center', height=0.5)
plt.yticks(ypos, industry)

for i, rect in enumerate(rects):
    ax.text(0.95 * rect.get_width(), rect.get_y() + rect.get_height() / 2.0, str(fluctuations[i]) + 'km/h', ha='right', va='center')

plt.xlabel('풍속(km/h)')
plt.ylabel('일')
plt.title('8월 최고 풍속')
plt.show()



'''
- 8월 최고풍속 막대그래프(수직)
'''
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rc
from matplotlib import style
import pandas as pd

df = pd.read_csv("d:\data\\final_incheon_airport1.csv")
maxwind = list(df['maxwind'][df['month'] == 8])
month = range(1, 32)
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
style.use('ggplot')
industry = month
fluctuations = maxwind
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
pos = np.arange(31)
rects = plt.bar(pos, fluctuations, align='center', width=0.5)
plt.xticks(pos, industry)
for i, rect in enumerate(rects):
    ax.text(rect.get_x() + rect.get_width() / 2.0, 0.95 * rect.get_height(), str(fluctuations[i]) + '%', ha='center')
plt.ylabel('등락률')
plt.ylabel('풍속(km/h)')
plt.xlabel('일')
plt.title('8월 최고 풍속')
plt.show()

# Malgun Gothic













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




