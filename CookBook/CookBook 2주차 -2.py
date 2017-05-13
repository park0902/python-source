'''
--------------------------------------------------------------------------------------
- 3.11 임의의 요소 뽑기

문제 : 시퀀스에서 임의의 아이템을 고르거나 난수 생성 하기
--------------------------------------------------------------------------------------
'''

'''
--------------------------------------------------------------------------------------
- 시퀀스에서 임의의 아이템을 선택하려면 random.choice() 사용
--------------------------------------------------------------------------------------
'''

import random

values = [1, 2, 3, 4, 5, 6]

print(random.choice(values))
print(random.choice(values))
print(random.choice(values))
print(random.choice(values))

'''
--------------------------------------------------------------------------------------
- 임의의 아이템을 N개 뽑아서 사용하고 버릴 목적이라면 random.sample() 사용
--------------------------------------------------------------------------------------
'''

import random

values = [1, 2, 3, 4, 5, 6]

print(random.sample(values, 2))
print(random.sample(values, 2))
print(random.sample(values, 3))
print(random.sample(values, 3))

'''
--------------------------------------------------------------------------------------
- 단순히 시퀀스의 아이템을 무작위로 섞으려면 random.shuffle() 사용
--------------------------------------------------------------------------------------
'''

import random

values = [1, 2, 3, 4, 5, 6]
random.shuffle(values)

print(values)

'''
--------------------------------------------------------------------------------------
- 임의의 정수를 생성하려면 random.randint() 사용
--------------------------------------------------------------------------------------
'''

import random

print(random.randint(0, 10))

'''
--------------------------------------------------------------------------------------
- 0과 1 사이의 균등 부동 소수점 값을 생성하려면 random.random() 사용
--------------------------------------------------------------------------------------
'''

import random

print(random.random())

'''
--------------------------------------------------------------------------------------
- N비트로 표현된 정수를 만들기 위해서는 random.getrandbits() 사용
--------------------------------------------------------------------------------------
'''

import random

print(random.getrandbits(200))

'''
--------------------------------------------------------------------------------------
- random 모듈은 Mersenne Twister 알고리즘을 사용해 난수를 발생
  이 알고리즘은 정해진 것 이지만 random.seed() 함수로 시드 값 변경 가능
  
- 그외에 기능
--------------------------------------------------------------------------------------
'''

import random

random.seed()               # 시스템 시간이나 os.urandom() 시드
random.seed(12345)          # 주어진 정수형 시드
random.seed(b'bytedata')    # 바이트 데이터 시드

random.uniform()            # 균등 분포 숫자 계산
random.gauss()              # 정규식 분포 숫자 계산





'''
--------------------------------------------------------------------------------------
3.12 시간 단위 변환

문제 : 날짜를 초, 시간을 분 으로 시간 단위 변환 하기
--------------------------------------------------------------------------------------
'''

'''
--------------------------------------------------------------------------------------
- 단위 변환이나 단위가 다른 값에 대한 계산을 하려면 datetime 모듈 사용
  시간의 간격을 나타내기 위해서는 timedelta 인스턴스 생성
--------------------------------------------------------------------------------------
'''

from datetime import timedelta

a = timedelta(days=2, hours=6)
b = timedelta(hours=4.5)
c = a + b

print(c.days)
print(c.seconds)
print(c.seconds / 3600)
print(c.total_seconds() / 3600)

'''
--------------------------------------------------------------------------------------
- 특정 날짜와 시간을 표현하려면 datetime 인스턴스를 만들고 표준 수학 연산 하기
--------------------------------------------------------------------------------------
'''

from datetime import datetime

a = datetime(2012, 9, 23)
b = datetime(2012, 12, 21)
d = b -a
now = datetime.today()

print(a + timedelta(days=10))
print(d.days)
print(now)
print(now + timedelta(minutes=10))

'''
--------------------------------------------------------------------------------------
- 계산을 할 때는 datetime이 윤년을 인식한다
--------------------------------------------------------------------------------------
'''

from datetime import datetime

a = datetime(2012, 3, 1)
b = datetime(2012, 2, 28)
c = datetime(2013, 3, 1)
d = datetime(2013, 2, 28)

print(a-b)
print((a-b).days)
print((c-d).days)

'''
--------------------------------------------------------------------------------------
- 시간대(time zone)나, 퍼지 시계 범위(fuzzy time range), 공휴일 계산 등의 더욱 복잡한 날짜 계산이
  필요하다면 dateutil 모듈 사용
  
- 대부분의 비슷한 시간 계산은 dateutil.relativedelta() 함수 수행
--------------------------------------------------------------------------------------
'''

from dateutil.relativedelta import relativedelta

a = datetime(2012, 3, 1)
b = datetime(2012, 12, 21)
d = b - a

print(a + relativedelta(months=+1))
print(a + relativedelta(months=+4))
print(d)

d = relativedelta(b, a)

print(d)
print(d.months)
print(d.days)





'''
--------------------------------------------------------------------------------------
3.13 마지막 금요일 날짜 구하기

문제 : 한 주의 마지막에 나타난 날의 날짜를 구하는 일반적인 해결책 만들기
--------------------------------------------------------------------------------------
'''

'''
--------------------------------------------------------------------------------------
- datetime 모듈의 클래스와 함수 사용
--------------------------------------------------------------------------------------
'''

from datetime import datetime, timedelta

weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
            'Friday', 'Saturday', 'Sunday']

def get_previous_byday(dayname, start_date=None):
    if start_date is None:
        start_date = datetime.today()
    day_num = start_date.weekday()
    day_num_target = weekdays.index(dayname)
    days_ago = (7 + day_num - day_num_target) % 7
    if days_ago == 0:
        days_ago = 7
    target_date = start_date - timedelta(days=days_ago)
    return target_date

print(datetime.today())
print(get_previous_byday('Monday'))
print(get_previous_byday('Tuesday'))
print(get_previous_byday('Friday'))

'''
--------------------------------------------------------------------------------------
- 시작 날짜와 목표 날짜를 관련 있는 숫자 위치에 매핑으로 목표 일자가 나타나고 며칠이 지났는지 알기
--------------------------------------------------------------------------------------
'''

from datetime import datetime
from dateutil.relativedelta import relativedelta
from dateutil.rrule import *

d = datetime.now()

print(d)
print(d + relativedelta(weekday=FR))        # 다음 금요일
print(d + relativedelta(weekday=FR(-1)))    # 마지막 금요일

