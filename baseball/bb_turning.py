# import random
#
# class Game:
#     def __init__(self):
#         print('====================================================================================================')
#         print('== 선수단 구성')
#         print('====================================================================================================')
#         print('== 선수단 구성이 완료 되었습니다.\n')
#         self.throws_numbers()
#
#     def throws_numbers(self):
#         random_balls = set()
#         while True:
#             random_balls.add(random.randint(1, 40))  # 1 ~ 20 중에 랜덤 수를 출력
#             if len(random_balls) == 4:  # 생성된 ball 이 4개 이면(set 객체라 중복 불가)
#                 print(random_balls)
#                 return random_balls
#
#
# class Human(Game):
#     def __init__(self):
#         super().__init__()
#
# if __name__ == '__main__':
#     human = Human()


import urllib.request
from  bs4 import BeautifulSoup
import re


abc = str(input('팀이름을을 입력하세요\n두산,엘지,SK,한화,삼성,KIA,롯데,KT,NC 중 고르세요\n입력하세요:'))

team = {'두산':'https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=1&ie=utf8&query=%EC%9E%A0%EC%8B%A4+%EB%82%A0%EC%94%A8',
        '엘지':'https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=1&ie=utf8&query=%EC%9E%A0%EC%8B%A4+%EB%82%A0%EC%94%A8',
        'SK':'https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=1&ie=utf8&query=%EB%AC%B8%ED%95%99%EB%8F%99+%EB%82%A0%EC%94%A8',
        '삼성':'https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=1&ie=utf8&query=%EC%97%B0%ED%98%B8%EB%8F%99+%EB%82%A0%EC%94%A8',
        '한화':'https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=1&ie=utf8&query=%EB%B6%80%EC%82%AC%EB%8F%99+%EB%82%A0%EC%94%A8',
        'KIA':'https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=1&ie=utf8&query=%EC%9E%84%EB%8F%99+%EB%82%A0%EC%94%A8',
        '롯데':'https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=1&ie=utf8&query=%EC%82%AC%EC%A7%81%EB%8F%99+%EB%82%A0%EC%94%A8',
        'KT':'https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=1&ie=utf8&query=%EC%A1%B0%EC%9B%90%EB%8F%99+%EB%82%A0%EC%94%A8',
        'NC':'https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=1&ie=utf8&query=%EC%96%91%EB%8D%95%EB%8F%99+%EB%82%A0%EC%94%A8'}

a = team[abc]

# binary = 'D:\chromedriver/chromedriver.exe'
# browser = webdriver.Chrome(binary)
list_url= a
url = urllib.request.Request(list_url)
res = urllib.request.urlopen(url).read().decode('utf-8')
soup = BeautifulSoup(res, 'html.parser')
b = soup.find('div',class_='contents03')
c = b.find_all('div',class_='w_now2')[0]
d = c.find('img')
print(d)
print('===='*10)
e = d.text.split()
print(e)
f = re.sub('[1-9,℃]','',e[2])
print(abc,'구장 날씨는',e[2])
weather = f[1:]
print(weather)
if weather == '맑음':
    print('6회까지만 경기 하겠습니다.')
else:
    print('9회까지 정상 경기 하겠습니다.')
# browser.quit()
print('===='*10)

