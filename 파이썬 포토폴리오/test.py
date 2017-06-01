# 폰트 확인 방법
# import matplotlib as mpl
#
# a = set(sorted([f.name for f in mpl.font_manager.fontManager.ttflist]))
#
# for i in a:
#     print(i)


# 파일 경로
# d:\data\\final_incheon_airport1.csv
csv_file = input('파일경로와 파일명 입력 하시오: ')
question = int(input('보고싶은 시각화 번호를 선택 하시오\n'
                     '1. 카테고리 그래프 \n'
                     '2. 월별 카테고리 그래프 \n'
                     '3. 여름 카테고리별 그래프 \n'
                     '4. 겨울 카테고리별 그래프 \n'
                     '5. 카테고리별 바 그래프(수평) \n'
                     '6. 카테고리별 바 그래프(수직) \n'
                     '7. 카테고리별 산점도 그래프 \n'
                     '8. (온도-이슬점) & 가시도 관계 그래프 \n'
                     '9. 가시도 & (온도-이슬점) 관계 산포도 및 선형 회귀 그래프 \n'))


class Visualization:
    # 카테고리에 해당하는 선그래프 출력
    def RKSI_2016_category(self):
        if question == 1:
            category = input('카테고리 입력 \n'
                             '- mean(평균기온) \n'
                             '- max(최고기온) \n'
                             '- min(최소기온) \n'
                             '- dew(이슬점) \n'
                             '- wind(평균풍속) \n'
                             '- maxwind(최고풍속) \n'
                             '- vis(가시도) \n')
            import matplotlib.pyplot as plt
            import pandas as pd
            import matplotlib as mpl
            mpl.rc('font', family='Malgun Gothic')
            mpl.rc('axes', unicode_minus=False)

            fig = plt.figure(); ax = fig.add_subplot(1,1,1)
            df = pd.read_csv(csv_file)
            max = list(df[category])
            month =range(1, 367)
            plt.plot(month, max, label=category, c="y")
            ax.set_xticks([0,31,60,91,121,151,181,212,243,273,304,334.366])
            ax.set_xticklabels(['January','February','March','April','May','Jun','July','August','September',
                                         'October','November','December'], rotation=30)
            ax.set_title(u'월별 '+category)
            plt.xlabel('Months')
            label = {'mean': 'Temperauture(°C)', 'max': 'Temperauture(°C)', 'min': 'Temperauture(°C)',
                     'dew': 'Temperauture(°C)', 'wind': 'Wind Speed(km/h)', 'maxwind': 'Wind Speed(km/h)', 'vis': 'Visibility(km)'}
            plt.ylabel(u'label[category]')
            ax.legend(loc='best')
            plt.show()


    # 카테고리에 해당하는 선그래프를 월별로 출력
    def RKSI_2016_monthly(self):
        if question == 2:
            category = input('카테고리 입력 \n'
                             '- mean(평균기온) \n'
                             '- max(최고기온) \n'
                             '- min(최소기온) \n'
                             '- dew(이슬점) \n'
                             '- wind(평균풍속) \n'
                             '- maxwind(최고풍속) \n'
                             '- vis(가시도) \n')
            import matplotlib.pyplot as plt
            import pandas as pd
            import matplotlib as mpl
            mpl.rc('font', family='Malgun Gothic')
            mpl.rc('axes', unicode_minus=False)

            df = pd.read_csv(csv_file)

            month1 = range(1, 30)
            month2 = range(1, 31)
            month3 = range(1, 32)

            re1 = list(df[category][df['month'] == 1]);re2 = list(df[category][df['month'] == 2]);re3 = list(df[category][df['month'] == 3])
            re4 = list(df[category][df['month'] == 4]);re5 = list(df[category][df['month'] == 5]);re6 = list(df[category][df['month'] == 6])
            re7 = list(df[category][df['month'] == 7]);re8 = list(df[category][df['month'] == 8]);re9 = list(df[category][df['month'] == 9])
            re10 = list(df[category][df['month'] == 10]);re11 = list(df[category][df['month'] == 11]);re12 = list(df[category][df['month'] == 12])

            plt.xlim(1, 32)
            plt.ylim(-20, 40)

            label = {'mean': u'온도(°C)', 'max': u'온도(°C)', 'min': u'온도(°C)',
                     'dew': u'온도(°C)', 'wind': u'풍속(km/h)', 'maxwind': u'풍속(km/h)',
                     'vis': u'가시도(km)'}
            title = {'mean': u'평균온도', 'max': u'최고온도', 'min': u'최저온도',
                     'dew': u'이슬점', 'wind': u'평균풍속', 'maxwind': u'최고풍속',
                     'vis': u'가시도'}

            plt.subplot(341);plt.plot(month3, re1);plt.title(u'1월 '+title[category]);plt.xlabel(u'일');plt.ylabel(label[category])
            plt.subplot(342);plt.plot(month1, re2);plt.title(u'2월 '+title[category]);plt.xlabel(u'일');plt.ylabel(label[category])
            plt.subplot(343);plt.plot(month3, re3);plt.title(u'3월 '+title[category]);plt.xlabel(u'일');plt.ylabel(label[category])
            plt.subplot(344);plt.plot(month2, re4);plt.title(u'4월 '+title[category]);plt.xlabel(u'일');plt.ylabel(label[category])
            plt.subplot(345);plt.plot(month3, re5);plt.title(u'5월 '+title[category]);plt.xlabel(u'일');plt.ylabel(label[category])
            plt.subplot(346);plt.plot(month2, re6);plt.title(u'6월 '+title[category]);plt.xlabel(u'일');plt.ylabel(label[category])
            plt.subplot(347);plt.plot(month3, re7);plt.title(u'7월 '+title[category]);plt.xlabel(u'일');plt.ylabel(label[category])
            plt.subplot(348);plt.plot(month3, re8);plt.title(u'8월 '+title[category]);plt.xlabel(u'일');plt.ylabel(label[category])
            plt.subplot(349);plt.plot(month2, re9);plt.title(u'9월 '+title[category]);plt.xlabel(u'일');plt.ylabel(label[category])
            plt.subplot(3, 4, 10);plt.plot(month3, re10);plt.title(u'10월 '+title[category]);plt.xlabel(u'일');plt.ylabel(label[category])
            plt.subplot(3, 4, 11);plt.plot(month2, re11);plt.title(u'11월 '+title[category]);plt.xlabel(u'일');plt.ylabel(label[category])
            plt.subplot(3, 4, 12);plt.plot(month3, re12);plt.title(u'12월 '+title[category]);plt.xlabel(u'일');plt.ylabel(label[category])
            plt.tight_layout()

            plt.show()


    # 여름(6,7,8월)의 카테고리별 선그래프 출력
    def RKSI_2016_summer_category(self):
        if question == 3:
            category = input('카테고리 입력 \n'
                             '- mean(평균기온) \n'
                             '- max(최고기온) \n'
                             '- min(최소기온) \n'
                             '- dew(이슬점) \n'
                             '- wind(평균풍속) \n'
                             '- maxwind(최고풍속) \n'
                             '- vis(가시도) \n')
            import matplotlib.pyplot as plt
            import pandas as pd

            df = pd.read_csv(csv_file)

            month1 = range(1, 30)
            month2 = range(1, 31)
            month3 = range(1, 32)

            re6 = list(df[category][df['month'] == 6])
            re7 = list(df[category][df['month'] == 7])
            re8 = list(df[category][df['month'] == 8])

            plt.plot(month2, re6, label='Jun', c="b", lw=1, ls=":", marker="D")
            plt.hold(True)
            plt.plot(month3, re7, label='July', c="g", lw=1, ls=":", marker="s")
            plt.hold(True)
            plt.plot(month3, re8, label='August', c="y", lw=1, ls=":", marker="o")
            plt.hold(True)

            label = {'mean': 'Temperauture(°C)', 'max': 'Temperauture(°C)', 'min': 'Temperauture(°C)',
                     'dew': 'Temperauture(°C)', 'wind': 'Wind Speed(km/h)', 'maxwind': 'Wind Speed(km/h)',
                     'vis': 'Visibility(km)'}

            plt.title('Summer '+category)
            plt.xlabel('Days')
            plt.ylabel(label[category])
            plt.legend(loc=3)

            plt.show()


    # 겨울(12,1,2월)의 카테고리별 선그래프 출력
    def RKSI_2016_winter_category(self):
        if question == 4:
            category = input('카테고리 입력 \n'
                             '- mean(평균기온) \n'
                             '- max(최고기온) \n'
                             '- min(최소기온) \n'
                             '- dew(이슬점) \n'
                             '- wind(평균풍속) \n'
                             '- maxwind(최고풍속) \n'
                             '- vis(가시도) \n')
            import matplotlib.pyplot as plt
            import pandas as pd

            df = pd.read_csv(csv_file)

            month1 = range(1, 30)
            month2 = range(1, 31)
            month3 = range(1, 32)

            re12 = list(df[category][df['month'] == 12])
            re1 = list(df[category][df['month'] == 1])
            re2 = list(df[category][df['month'] == 2])

            plt.plot(month3, re12, label='Jun', c="b", lw=1, ls=":", marker="D")
            plt.hold(True)
            plt.plot(month3, re1, label='July', c="g", lw=1, ls=":", marker="s")
            plt.hold(True)
            plt.plot(month1, re2, label='August', c="y", lw=1, ls=":", marker="o")
            plt.hold(True)

            label = {'mean': 'Temperauture(°C)', 'max': 'Temperauture(°C)', 'min': 'Temperauture(°C)',
                     'dew': 'Temperauture(°C)', 'wind': 'Wind Speed(km/h)', 'maxwind': 'Wind Speed(km/h)',
                     'vis': 'Visibility(km)'}

            plt.title('Winter '+category)
            plt.xlabel('Days')
            plt.ylabel(label[category])
            plt.legend(loc=3)

            plt.show()


    # 월의 카테고리별 바 그래프(수평)
    def RKSI_2016_bar_hor(self):
        if question == 5:
            category = input('카테고리 입력 \n'
                             '- mean(평균기온) \n'
                             '- max(최고기온) \n'
                             '- min(최소기온) \n'
                             '- dew(이슬점) \n'
                             '- wind(평균풍속) \n'
                             '- maxwind(최고풍속) \n'
                             '- vis(가시도) \n')
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib import font_manager, rc
            from matplotlib import style
            import pandas as pd
            m = int(input('월 입력 : '))
            label = {'mean': 'Temperauture(°C)', 'max': 'Temperauture(°C)', 'min': 'Temperauture(°C)',
                     'dew': 'Temperauture(°C)', 'wind': 'Wind Speed(km/h)', 'maxwind': 'Wind Speed(km/h)',
                     'vis': 'Visibility(km)'}
            title = {'mean': u'평균온도', 'max': u'최고온도', 'min': u'최저온도',
                     'dew': u'이슬점', 'wind': u'평균풍속', 'maxwind': u'최고풍속',
                     'vis': u'가시도'}

            if m == 1 or m == 3 or m == 5 or m == 7 or m == 8 or m == 10 or m == 12:
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
                    ax.text(rect.get_x() + rect.get_width() / 2.0, 0.95 * rect.get_height(), str(fluctuations[i]) ,
                            ha='center')
                plt.ylabel('등락률')
                plt.ylabel('풍속(km/h)')
                plt.xlabel('일')
                plt.title('8월 최고 풍속')
                plt.show()

            elif m == 4 or m == 6 or m == 9 or m == 11:
                import matplotlib.pyplot as plt
                import numpy as np
                from matplotlib import font_manager, rc
                from matplotlib import style
                import pandas as pd

                df = pd.read_csv("d:\data\\final_incheon_airport1.csv")
                maxwind = list(df['maxwind'][df['month'] == 8])
                month = range(1, 31)
                font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
                rc('font', family=font_name)
                style.use('ggplot')
                industry = month
                fluctuations = maxwind
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111)
                pos = np.arange(30)
                rects = plt.bar(pos, fluctuations, align='center', width=0.5)
                plt.xticks(pos, industry)
                for i, rect in enumerate(rects):
                    ax.text(rect.get_x() + rect.get_width() / 2.0, 0.95 * rect.get_height(), str(fluctuations[i]) + '%',
                            ha='center')
                plt.ylabel('등락률')
                plt.ylabel('풍속(km/h)')
                plt.xlabel('일')
                plt.title('8월 최고 풍속')
                plt.show()
            else:
                import matplotlib.pyplot as plt
                import numpy as np
                from matplotlib import font_manager, rc
                from matplotlib import style
                import pandas as pd

                df = pd.read_csv("d:\data\\final_incheon_airport1.csv")
                maxwind = list(df['maxwind'][df['month'] == 8])
                month = range(1, 30)
                font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
                rc('font', family=font_name)
                style.use('ggplot')
                industry = month
                fluctuations = maxwind
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111)
                pos = np.arange(29)
                rects = plt.bar(pos, fluctuations, align='center', width=0.5)
                plt.xticks(pos, industry)
                for i, rect in enumerate(rects):
                    ax.text(rect.get_x() + rect.get_width() / 2.0, 0.95 * rect.get_height(), str(fluctuations[i]) + '%',
                            ha='center')
                plt.ylabel('등락률')
                plt.ylabel('풍속(km/h)')
                plt.xlabel('일')
                plt.title('8월 최고 풍속')
                plt.show()


    # 월의 카테고리별 바 그래프(수직)
    def RKSI_2016_bar_ver(self):
            if question == 6:
                category = input('카테고리 입력 \n'
                                 '- mean(평균기온) \n'
                                 '- max(최고기온) \n'
                                 '- min(최소기온) \n'
                                 '- dew(이슬점) \n'
                                 '- wind(평균풍속) \n'
                                 '- maxwind(최고풍속) \n'
                                 '- vis(가시도) \n')
                import pandas as pd
                m = int(input('월 입력 : '))
                if m == 1 or m == 3 or m ==5 or m==7 or m==8 or m==10 or m==12:
                    df = pd.read_csv(csv_file)
                    re = list(df[category][df['month'] == m])
                    month = range(1,32)
                    df = pd.DataFrame(re, index=month, columns=pd.Index([category]))
                    df.plot(kind='bar', color='b', alpha=0.5)

                elif m ==4 or m==6 or m==9 or m==11:
                    df = pd.read_csv(csv_file)
                    re = list(df[category][df['month'] == m])

                    month = range(1, 31)

                    df = pd.DataFrame(re, index=month, columns=pd.Index([category]))
                    df.plot(kind='bar', color='b', alpha=0.5)

                else:
                    df = pd.read_csv(csv_file)
                    re = list(df[category][df['month'] == m])

                    month = range(1, 30)

                    df = pd.DataFrame(re, index=month, columns=pd.Index([category]))
                    df.plot(kind='bar', color='b', alpha=0.5)


    # 카테고리별 산포도 그리기
    def Scatter(self):
        if question == 7:
            category1 = input('X축 카테고리 입력 \n'
                              '- mean(평균기온) \n'
                              '- max(최고기온) \n'
                              '- min(최소기온) \n'
                              '- dew(이슬점) \n'
                              '- wind(평균풍속) \n'
                              '- maxwind(최고풍속) \n'
                              '- vis(가시도) \n')
            category2 = input('Y 축 카테고리 입력 \n'
                              '- mean(평균기온) \n'
                              '- max(최고기온) \n'
                              '- min(최소기온) \n'
                              '- dew(이슬점) \n'
                              '- wind(평균풍속) \n'
                              '- maxwind(최고풍속) \n'
                              '- vis(가시도) \n')
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.read_csv(csv_file)
        re1 = list(df[category1])
        re2 = list(df[category2])
        plt.scatter(re1, re2)
        plt.title('2016 RKSI ',category1,' & ',category2)
        plt.ylabel(category2)
        plt.xlabel(category1)



    # (온도-이슬점) & 가시도 관계 그래프
    def Min_Dew_Vis(self):
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np

        df = pd.read_csv("d:\data\\final_incheon_airport1.csv")

        year = range(1, 367)

        mean = np.array(df['mean'])
        dew = np.array(df['dew'])
        re = list(mean - dew)
        vis = list(df['vis'])

        plt.plot(year, re, label='mean-dew', )
        plt.hold(True)
        plt.plot(year, vis, label='vis')
        plt.hold(False)

        plt.title('2016 RKSI (Mean-Dew Point) & Visualization')
        plt.yticks([0, 10, 20])
        plt.xlabel('Days')
        plt.ylabel('Temperature & Visualization')

        plt.legend(loc=2)

        plt.show()

    '''
    - (온도 - 이슬점) 과 가시도 와의 관계

      온도가 이슬점에 가까워 지거나 이슬점이 현재 온도까지 상승하게되면
      안개가 끼고, 온도와 이슬점이 같아지면 100% 안개가 끼어 가시도(visibility)가 나빠진다
      즉, 이슬점과 온도와 큰 차이를 나타내면 가시도 양호
    '''



    # 가시도 & (온도-이슬점) 관계 산포도 및 선형 회귀 그래프
    def LinearRegression(self):
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.read_csv(csv_file)
        mean = np.array(df['mean'])
        dew = np.array(df['dew'])
        re = mean - dew
        vis = np.array(df['vis'])

        A = np.vstack([re, np.ones(len(re))]).T

        a, b = np.linalg.lstsq(A, vis)[0]

        plt.plot(vis, re, 'o', label='data', markersize=8)
        plt.hold(True)
        plt.plot(a * re + b, re, 'r', label='Fitted Line')
        plt.hold(False)
        plt.legend()
        plt.show()

        plt.title('2016 RKSI Dew point & (Mean Temperature - Dew Point)')
        plt.ylabel('(Mean Temperature - Dew Point)')
        plt.xlabel('Visibility')

        print('가시도 = ', '(온도-이슬점) * ', a, ' + ', b)


        # 가시도 =  (온도-이슬점) *  0.851212922822  +  4.05691172106




v = Visualization()
if question == 1:
    v.RKSI_2016_category()
if question == 2:
    v.RKSI_2016_monthly()
if question == 3:
    v.RKSI_2016_summer_category()
if question == 4:
    v.RKSI_2016_winter_category()
if question == 5:
    v.RKSI_2016_bar_hor()
if question == 6:
    v.RKSI_2016_bar_ver()
if question == 7:
    v.Scatter()
if question == 8:
    v.Min_Dew_Vis()
if question == 9:
    v.LinearRegression()



