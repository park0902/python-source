'''
- 지역, 년도, 요소에 해당하는 숫자값을 기상청 html 에서 가져오는 기능
'''
# # -*- coding: utf-8 -*-
# from bs4 import BeautifulSoup
# from urllib.request import Request, urlopen
# import operator
# import time
#
#
# class KMACrawler:
#     FILE_PATH = 'D:\\kma\\'     # 기상 데이터 수집할 위치를 지정
#
#     def __init__(self):
#         self.location_list = {}     # 검색조건 3개중 지역의 데이터를 담을 변수
#         self.year_list = {}         # 검색조건중 연도(1960~2017) 데이터를 담을 변수
#         self.factor_list = {}       # 검색조건 3개중 요소(평균기온, 최고기온, ..)
#         self.crawling_list = {}     # 위의 3개지 조건으로 수집한 데이터
#                                     # (('136','2015, 12, ('안동(무), '2015', '상대습도'))
#
#         self.data = {}              # 실제 결과 데이터를 담는 딕셔너리 변수
#         # 백령도(유), 2001, 최고기온
#         # 3, 7, -1, 5, 7.0, 8.5...............
#
#         # 기상처 온도 확인하는 메인 url
#         self.default_url = 'http://www.kma.go.kr/weather/climate/past_table.jsp'
#
#         # 특정지역, 연도, 요소에 따른 데이터 조회하는 상세 url
#         self.crawled_url = 'http://www.kma.go.kr/weather/climate/past_table.jsp?stn={}&yy={}&obs={}'
#
#
#     # 지점, 연도, 요소에 데이터 가져오는 함수
#     def get_kma_data(self):
#         # 메인 url을 통해서 html 코드를 가져오는 부분
#         res = urlopen(Request(self.default_url)).read()
#         # print(res)
#
#         # res html 코드를 beautifulsoup로 검색 설정
#         soup = BeautifulSoup(res, 'html.parser')
#
#         # 지역에 관련한 html 코드만 가져오는 부분
#         location = soup.find('select', id='observation_select1')
#         # print(location)
#
#         # 연도에 관련한 html 코드만 가져오는 부분
#         year = soup.find('select', id='observation_select2')
#         # print(year)
#
#         # 평균 기온과 같은 요소에 관련된 html 코드만 가져오는 부분
#         factor = soup.find('select', id='observation_select3')
#         # print(factor)
#
#         for tag in location.find_all('option'):
#             if tag.text != '--------':      # 구분선 나오지 않게 해라!
#                 self.location_list[tag['value']] = tag.text
#                 # print(tag['value']) # 188
#                 # print(tag.text)     # 성산(무)
#
#         for tag in year.find_all('option'):
#             if tag.text != '--------':
#                 self.year_list[tag['value']] = tag.text
#                 # print(tag['value']) #  1961
#                 # print(tag.text)     #  1961
#
#         for tag in factor.find_all('option'):
#             if tag.text != '--------':
#                 self.factor_list[tag['value']] = tag.text
#                 # print(tag['value']) # 06
#                 # print(tag.text)     # 평균풍속
#
#         # print(self.location_list.items())
#         # print(self.year_list.items())
#         # print(self.factor_list.items())
#         for loc_key, loc_value in self.location_list.items():
#             for year_key, year_value in self.year_list.items():
#                 for fac_key, fac_value in self.factor_list.items():
#                     self.crawling_list[(loc_key, year_key, fac_key)] = (loc_value, year_value, fac_value)
#
#         # print(self.crawling_list)
#
#
# crawler = KMACrawler()
# crawler.get_kma_data()





'''
- 3개의 숫자값을 이용해서 상세 url을 완성해서 온도 데이터를 가져오는 기능 
'''
# # # -*- coding: utf-8 -*-
# from bs4 import BeautifulSoup
# from urllib.request import Request, urlopen
# import operator
# import time
#
# class KMACrawler:
#     FILE_PATH = 'D:\\kma\\'
#     def __init__(self):
#         self.location_list = {}
#         self.year_list = {}
#         self.factor_list = {}
#         self.crawling_list = {}
#         self.data = {}
#         self.default_url = 'http://www.kma.go.kr/weather/climate/past_table.jsp'
#         self.crawled_url = 'http://www.kma.go.kr/weather/climate/past_table.jsp?stn={}&yy={}&obs={}'
#         #self.play_crawling()
#
#
#     # 크롤링 수행하는 메인 함수
#     def play_crawling(self):
#         print('크롤링을 위한 데이터를 수집 중입니다...')
#         self.crawling_list = {('108', '1986', '35'): ('서울(유)', '1986', '일조시간')}
#         print('크롤링을 위한 데이터 수집 완료 !!!')
#         print('크롤링을 시작합니다...')
#         for key, value in sorted(self.crawling_list.items(), key=operator.itemgetter(0)):
#             # 상세 url 완성해서 기상청 웹서버에 요청해서 html 받아옴
#             res = urlopen(Request(self.crawled_url.format(key[0], key[1], key[2]))).read()
#
#             soup = BeautifulSoup(res, 'html.parser')
#             print('현재 키워드 : {}, {}, {}'.format(*value))
#             for tr_tag in soup.find('table', class_='table_develop').find('tbody').find_all('tr'):
#                 # print(tr_tag)
#                 if self.data.get(value) is None:
#                     self.data[value] = []
#                 self.data[value].append([td_tag.text for td_tag in tr_tag.find_all('td')])
#
#             print (self.data.items())
#             print('{}, {}, {} 에 대한 데이터 저장...'.format(*value))
#
#             self.data.clear()
#             print('저장 완료!!!\n\n')
#             time.sleep(2)
#         print('크롤링 완료 !!!')
#
# crawler = KMACrawler()
# crawler.play_crawling()


'''
-  data_to_file() 함수를 play_crawling() 함수를 실행되게해서 온도 데이터가 d 드라이브 밑에 kma 폴더에 저장!
'''
# # # -*- coding: utf-8 -*-
# from bs4 import BeautifulSoup
# from urllib.request import Request, urlopen
# import operator
# import time
#
# class KMACrawler:
#     FILE_PATH = 'D:\\kma\\'
#     def __init__(self):
#         self.location_list = {}
#         self.year_list = {}
#         self.factor_list = {}
#         self.crawling_list = {}
#         self.data = {}
#         self.default_url = 'http://www.kma.go.kr/weather/climate/past_table.jsp'
#         self.crawled_url = 'http://www.kma.go.kr/weather/climate/past_table.jsp?stn={}&yy={}&obs={}'
#         #self.play_crawling()
#
#     def data_to_file(self):
#         with open(KMACrawler.FILE_PATH + "kma_crawled.txt", "a", encoding="utf-8") as file:
#             file.write('======================================================\n')
#             for key, value in self.data.items():
#                 file.write('>> ' + key[0] + ', ' + key[1] + ', ' + key[2] + '\n')
#                 for v in value:
#                     file.write(','.join(v) + '\n')
#             file.write('======================================================\n\n')
#             file.close()
#
#
#     # 크롤링 수행하는 메인 함수
#     def play_crawling(self):
#         print('크롤링을 위한 데이터를 수집 중입니다...')
#         self.crawling_list = {('108', '1986', '35'): ('서울(유)', '1986', '일조시간')}
#         print('크롤링을 위한 데이터 수집 완료 !!!')
#         print('크롤링을 시작합니다...')
#         for key, value in sorted(self.crawling_list.items(), key=operator.itemgetter(0)):
#             # 상세 url 완성해서 기상청 웹서버에 요청해서 html 받아옴
#             res = urlopen(Request(self.crawled_url.format(key[0], key[1], key[2]))).read()
#
#             soup = BeautifulSoup(res, 'html.parser')
#             print('현재 키워드 : {}, {}, {}'.format(*value))
#             for tr_tag in soup.find('table', class_='table_develop').find('tbody').find_all('tr'):
#                 # print(tr_tag)
#                 if self.data.get(value) is None:
#                     self.data[value] = []
#                 self.data[value].append(['' if td_tag.text == '\xa0' else td_tag.text for td_tag in tr_tag.find_all('td') if td_tag.has_attr('scope') is False])
#
#             print (self.data.items())
#             print('{}, {}, {} 에 대한 데이터 저장...'.format(*value))
#
#             self.data_to_file()
#             self.data.clear()
#             print('저장 완료!!!\n\n')
#             time.sleep(2)
#         print('크롤링 완료 !!!')
#
# crawler = KMACrawler()
# crawler.play_crawling()







'''
- 기상청 크롤링
'''
# # -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import operator
import time

class KMACrawler:
    FILE_PATH = 'D:\\kma\\'
    def __init__(self):
        self.location_list = {}
        self.year_list = {}
        self.factor_list = {}
        self.crawling_list = {}
        self.data = {}
        self.default_url = 'http://www.kma.go.kr/weather/climate/past_table.jsp'
        self.crawled_url = 'http://www.kma.go.kr/weather/climate/past_table.jsp?stn={}&yy={}&obs={}'

    def data_to_file(self):
        with open(KMACrawler.FILE_PATH + "kma_crawled.txt", "a", encoding="utf-8") as file:
            file.write('======================================================\n')
            for key, value in self.data.items():
                file.write('>> ' + key[0] + ', ' + key[1] + ', ' + key[2] + '\n')
                for v in value:
                    file.write(','.join(v) + '\n')
            file.write('======================================================\n\n')
            file.close()


     # 지점, 연도, 요소에 데이터 가져오는 함수
    def get_kma_data(self):
        # 메인 url을 통해서 html 코드를 가져오는 부분
        res = urlopen(Request(self.default_url)).read()

        # res html 코드를 beautifulsoup로 검색 설정
        soup = BeautifulSoup(res, 'html.parser')

        # 지역에 관련한 html 코드만 가져오는 부분
        location = soup.find('select', id='observation_select1')

        # 연도에 관련한 html 코드만 가져오는 부분
        year = soup.find('select', id='observation_select2')

        # 평균 기온과 같은 요소에 관련된 html 코드만 가져오는 부분
        factor = soup.find('select', id='observation_select3')

        for tag in location.find_all('option'):
            if tag.text != '--------':      # 구분선 나오지 않게 해라!
                self.location_list[tag['value']] = tag.text
                # print(tag['value']) # 188
                # print(tag.text)     # 성산(무)

        for tag in year.find_all('option'):
            if tag.text != '--------':
                self.year_list[tag['value']] = tag.text
                # print(tag['value']) #  1961
                # print(tag.text)     #  1961

        for tag in factor.find_all('option'):
            if tag.text != '--------':
                self.factor_list[tag['value']] = tag.text
                # print(tag['value']) # 06
                # print(tag.text)     # 평균풍속

        for loc_key, loc_value in self.location_list.items():
            for year_key, year_value in self.year_list.items():
                for fac_key, fac_value in self.factor_list.items():
                    self.crawling_list[(loc_key, year_key, fac_key)] = (loc_value, year_value, fac_value)


    # 크롤링 수행하는 메인 함수
    def play_crawling(self):
        print('크롤링을 위한 데이터를 수집 중입니다...')
        self.get_kma_data()
        print('크롤링을 위한 데이터 수집 완료 !!!')
        print('크롤링을 시작합니다...')
        for key, value in sorted(self.crawling_list.items(), key=operator.itemgetter(0)):
            # 상세 url 완성해서 기상청 웹서버에 요청해서 html 받아옴
            res = urlopen(Request(self.crawled_url.format(key[0], key[1], key[2]))).read()

            soup = BeautifulSoup(res, 'html.parser')
            print('현재 키워드 : {}, {}, {}'.format(*value))
            for tr_tag in soup.find('table', class_='table_develop').find('tbody').find_all('tr'):
                # print(tr_tag)
                if self.data.get(value) is None:
                    self.data[value] = []
                self.data[value].append(['' if td_tag.text == '\xa0' else td_tag.text for td_tag in tr_tag.find_all('td') if td_tag.has_attr('scope') is False])

            print('{}, {}, {} 에 대한 데이터 저장...'.format(*value))

            self.data_to_file()
            self.data.clear()
            print('저장 완료!!!\n\n')
            time.sleep(2)
        print('크롤링 완료 !!!')

crawler = KMACrawler()
crawler.play_crawling()








