csv_file = input('파일경로와 파일명 입력: ')
question = int(input('보고싶은 시각화 선택(1. 카테고리 그래프 2. 월별 카테고리 그래프 3. 여름 카테고리별 그래프 4. 겨울 카테고리별 그래프 '
                     '5. 카테고리별 바 그래프(수평) 6. 카테고리별 바 그래프(수직) 7. 가시도 & (온도-이슬점) 관계 산포도 및 선형 회귀 그래프'))
category = input('카테고리 입력(mean, max, min, dew, wind, maxwind, vis)')

# d:\data\\final_incheon_airport1.csv

def RKSI_2016_category():
    import matplotlib.pyplot as plt
    import pandas as pd
    fig = plt.figure(); ax = fig.add_subplot(1,1,1)
    df = pd.read_csv(csv_file)
    max = list(df[category])
    month =range(1, 367)
    plt.plot(month, max, label=category, c="y")
    ax.set_xticks([0,31,60,91,121,151,181,212,243,273,304,334.366])
    ax.set_xticklabels(['January','February','March','April','May','Jun','July','August','September',
                                 'October','November','December'], rotation=30)
    ax.set_title('RKSI Monthly '+category+'(2016)')
    plt.xlabel('Months')
    label = {'mean': 'Temperauture(°C)', 'max': 'Temperauture(°C)', 'min': 'Temperauture(°C)',
             'dew': 'Temperauture(°C)', 'wind': 'Wind Speed(km/h)', 'maxwind': 'Wind Speed(km/h)', 'vis': 'Visibility(km)'}
    plt.ylabel(label[category])
    ax.legend(loc='best')
    plt.show()


def RKSI_2016_monthly():
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.read_csv(csv_file)

        month1 = range(1, 30)
        month2 = range(1, 31)
        month3 = range(1, 32)

        re1 = list(df[category][df['month'] == 1])
        re2 = list(df[category][df['month'] == 2])
        re3 = list(df[category][df['month'] == 3])
        re4 = list(df[category][df['month'] == 4])
        re5 = list(df[category][df['month'] == 5])
        re6 = list(df[category][df['month'] == 6])
        re7 = list(df[category][df['month'] == 7])
        re8 = list(df[category][df['month'] == 8])
        re9 = list(df[category][df['month'] == 9])
        re10 = list(df[category][df['month'] == 10])
        re11 = list(df[category][df['month'] == 11])
        re12 = list(df[category][df['month'] == 12])

        plt.xlim(1, 32)
        plt.ylim(-20, 40)

        label = {'mean': 'Temperauture(°C)', 'max': 'Temperauture(°C)', 'min': 'Temperauture(°C)',
                 'dew': 'Temperauture(°C)', 'wind': 'Wind Speed(km/h)', 'maxwind': 'Wind Speed(km/h)',
                 'vis': 'Visibility(km)'}

        plt.subplot(341);plt.plot(month3, re1);plt.title('January '+category);plt.xlabel('Days');plt.ylabel(label[category])
        plt.subplot(342);plt.plot(month1, re2);plt.title('February '+category);plt.xlabel('Days');plt.ylabel(label[category])
        plt.subplot(343);plt.plot(month3, re3);plt.title('March '+category);plt.xlabel('Days');plt.ylabel(label[category])
        plt.subplot(344);plt.plot(month2, re4);plt.title('April '+category);plt.xlabel('Days');plt.ylabel(label[category])
        plt.subplot(345);plt.plot(month3, re5);plt.title('May '+category);plt.xlabel('Days');plt.ylabel(label[category])
        plt.subplot(346);plt.plot(month2, re6);plt.title('Jun '+category);plt.xlabel('Days');plt.ylabel(label[category])
        plt.subplot(347);plt.plot(month3, re7);plt.title('July '+category);plt.xlabel('Days');plt.ylabel(label[category])
        plt.subplot(348);plt.plot(month3, re8);plt.title('August '+category);plt.xlabel('Days');plt.ylabel(label[category])
        plt.subplot(349);plt.plot(month2, re9);plt.title('September '+category);plt.xlabel('Days');plt.ylabel(label[category])
        plt.subplot(3, 4, 10);plt.plot(month3, re10);plt.title('October '+category);plt.xlabel('Days');plt.ylabel(label[category])
        plt.subplot(3, 4, 11);plt.plot(month2, re11);plt.title('November '+category);plt.xlabel('Days');plt.ylabel(label[category])
        plt.subplot(3, 4, 12);plt.plot(month3, re12);plt.title('December '+category);plt.xlabel('Days');plt.ylabel(label[category])
        plt.tight_layout()

        plt.show()


def RKSI_2016_summer_category():
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


def RKSI_2016_winter_category():
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv("d:\data\\final_incheon_airport1.csv")

    month1 = range(1, 30)
    month2 = range(1, 31)
    month3 = range(1, 32)

    re12 = list(df['max'][df['month'] == 12])
    re1 = list(df['max'][df['month'] == 1])
    re2 = list(df['max'][df['month'] == 2])

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


def RKSI_2016_bar_hor():
    if question == 5:
        import pandas as pd
        m = int(input('월 입력 : '))
        if m == 1 or m == 3 or m == 5 or m == 7 or m == 8 or m == 10 or m == 12:
            df = pd.read_csv(csv_file)
            re = list(df[category][df['month'] == m])
            month = range(1, 32)
            df = pd.DataFrame(re, index=month, columns=pd.Index([category]))
            df.plot(kind='barh', color='b', alpha=0.5)

        elif m == 4 or m == 6 or m == 9 or m == 11:
            df = pd.read_csv(csv_file)
            re = list(df[category][df['month'] == m])

            month = range(1, 31)

            df = pd.DataFrame(re, index=month, columns=pd.Index([category]))
            df.plot(kind='barh', color='b', alpha=0.5)

        else:
            df = pd.read_csv(csv_file)
            re = list(df[category][df['month'] == m])

            month = range(1, 30)

            df = pd.DataFrame(re, index=month, columns=pd.Index([category]))
            df.plot(kind='barh', color='b', alpha=0.5)

def RKSI_2016_bar_ver():
        if question == 6:
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


def linearRegression():
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






if question == 1:
    RKSI_2016_category()
if question == 2:
    RKSI_2016_monthly()
if question == 3:
    RKSI_2016_summer_category()
if question == 4:
    RKSI_2016_winter_category()
if question == 5:
    RKSI_2016_bar_hor()
if question == 6:
    RKSI_2016_bar_ver()
if question == 7:
    linearRegression()
