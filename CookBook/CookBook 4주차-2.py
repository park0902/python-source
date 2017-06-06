'''
--------------------------------------------------------------------------------------
6.5 딕셔너리를 XML 로 바꾸기

문제 : 파이썬 딕셔너리 데이터를 받아서 XML 로 바꾸기
--------------------------------------------------------------------------------------
'''

'''
--------------------------------------------------------------------------------------
- xml.etree.ElementTree 라이브러리는 파싱에 일반적으로 사용하지만 XML 문서를 생성할 때 사용
--------------------------------------------------------------------------------------
'''

from xml.etree.ElementTree import Element
from xml.etree.ElementTree import tostring

def dict_to_xml(tag, d):
    '''
    간단한 dict를 xml 로 변환하기
    '''
    elem = Element(tag)
    for key, val in d.items():
        child = Element(key)
        child.text = str(val)
        elem.append(child)
    return elem

s = {'name' : 'GOOD', 'share' : 100, 'price' : 490.1}
e = dict_to_xml('stock', s)

# Element 인스턴스
print(e)

# 바이트 문자열
print(tostring(e))

# 요소에 속성 넣기
e.set('_id', '1234')
print(tostring(e))

'''
--------------------------------------------------------------------------------------
- XML을 생성할 때 단순히 문자열을 사용 
--------------------------------------------------------------------------------------
'''

from xml.sax.saxutils import escape, unescape

def dict_to_xml_str(tag, d):
    '''
    간단한 dict 를 xml로 변환하기
    '''
    parts = ['<{}>'.format(tag)]
    for key, val in d.items():
        parts.append('<{0}>{1}</{0}>'.format(key, val))
    parts.append('</{}>'.format(tag))
    return ''.join(parts)

d = {'name' : '<spam>'}

# 문자열 생성
dict_to_xml_str('item', d)

# 올바른 XML 생성
e = dict_to_xml_str('item', d)
print(tostring(e))

# 문자를 수동으로 이스케이핑하기
print(escape('<spam>'))

print(unescape())

'''
--------------------------------------------------------------------------------------
- 올바른 출력을 만드는 것 외에도 문자열 대신 Element 인스턴스를 만드는 것이 좋은 이유는
  이들을 더 쉽게 합쳐 큰 문서를 만들 수 있기 때문이다
  
- Element 인스턴스는 XML 파싱에 대한 염려 없이 여러 방법으로 처리할 수 있다
--------------------------------------------------------------------------------------
'''





'''
--------------------------------------------------------------------------------------
6.6 XML 파싱, 수정, 저장

문제 : XML 문서를 읽고, 수정하고 수정 내용을 XML에 반영하기
--------------------------------------------------------------------------------------
'''

'''
--------------------------------------------------------------------------------------
- xml.etree.ElementTree 모듈 사용(pred.xml)
--------------------------------------------------------------------------------------
'''

from xml.etree.ElementTree import parse, Element

doc = parse('pred.xml')
root = doc.getroot()

print(root)

# 요소 몇 개 제거하기
root.remove(root.find('sti'))
root.remove(root.find('cr'))

# <nm> ... </nm> 뒤에 요소 몇 개 삽입하기
root.getchildren().idex(root.find('nm'))
e = Element('spam')
e.text = 'This is a test'
root.insert(2, e)

# 파일에 쓰기
doc.write('newpred.xml', xml_declaration=True)

'''
--------------------------------------------------------------------------------------
- 어떤 요소를 제거하면 부모의 remove() 메소를 사용해 바로 위에 있는 부모로부터 제거된다

- 새로운 요소를 추가하면 부모에 대해서도 insert() 와 append() 메소드를 사용

- 모든 요소는 element[i] 또는 element[i:j] 와 같이 인덱스와 슬라이스 명령으로도 접근
--------------------------------------------------------------------------------------
'''





'''
--------------------------------------------------------------------------------------
6.7 네임스페이스로 XML 문서 파싱

문제 : XML 문서를 파싱할 때 XML 네임스페이스(namespace) 사용 
--------------------------------------------------------------------------------------
'''

# 동작하는 쿼리
doc.findall('author')
doc.find('content')

# 네임스페이스 관련 쿼리(동작하지 않음)
doc.find('content/html')

# 조건에 맞는 경우에만 동작
doc.find('content/{http://www.w3.org/1999/xhtml}html')

# 동작하지 않음
doc.findtext('content/{http://wwww.w3.org/1999/xhtml}html/head/title')

# 조건에 일치함
doc.findtext('content/{http://www.w3.org/1999/xhtml}html')

'''
--------------------------------------------------------------------------------------
- 유클리드 클래스로 네임스페이스 감싸기
--------------------------------------------------------------------------------------
'''

class XMLNamespaces:
    def __init__(self, **kwargs):
        self.namespaces = {}
        for name, uri in kwargs.items():
            self.register(name, uri)

    def register(self, name, uri):
        self.namespaces[name] = '{'+uri+'}'

    def __call__(self, path):
        return path.format_map(self.namespaces)

ns = XMLNamespaces(html='http://www.w3.org/1999/xhtml')

doc.find(ns('content/{html}html'))

'''
--------------------------------------------------------------------------------------
- iterparse() 함수를 사용한다면 네임스페이스 처리의 범위에 대해서 정보 얻기
--------------------------------------------------------------------------------------
'''

from xml.etree.ElementTree import iterparse

for evt, elem in iterparse('ns2.xml', ('end', 'start-ns', 'end-ns')):
    print(evt, elem)





'''
--------------------------------------------------------------------------------------
6.8 관계형 데이터베이스 작업

문제 : 관계형 데이터베이스에 선택, 삽입, 행 삭제(select, insert, delete row) 등의 작업 수행
--------------------------------------------------------------------------------------
'''

'''
--------------------------------------------------------------------------------------
- 파이썬에서 데이터 행을 나타내는 표준은 튜플 시퀀스
--------------------------------------------------------------------------------------
'''

stocks = [('GOOD', 100, 490.1), ('AAPL', 50, 545.75),
          ('FB', 150, 7.45), ('HPQ', 75, 33.2)]

'''
--------------------------------------------------------------------------------------
- 데이터베이스를 연결
  connect() 함수에 데이터베이스 이름, 호스트 이름, 사용자 이름, 암호 등 필요한 정보 넣기
--------------------------------------------------------------------------------------
'''

import sqlite3

stocks = [('GOOD', 100, 490.1), ('AAPL', 50, 545.75),
          ('FB', 150, 7.45), ('HPQ', 75, 33.2)]

db = sqlite3.connect('database.db')

# 커서를 만든 후에 SQL 쿼리를 실행할 수 있다
c = db.cursor()
c.execute('create table portfolio (symbol text, share integer, price real)')
db.commit()

# 데이터에 행의 시퀀스 삽입
c.executemany('insert into portfolio values (?,?,?)', stocks)
db.commit()

# 쿼리 수행
for row in db.execute('select * from portfolio'):
    print(row)

# 사용자가 입력한 파라미터를 받는 쿼리를 수행하려면 ?를 사용해 파라미터를 이스케이핑
min_price = 100

for row in db.execute('select * from portfolio where price >= ?', (min_price,)):
    print(row)





'''
--------------------------------------------------------------------------------------
6.9 16진수 인코딩, 디코딩

문제 : 문자열로 16진수를 바이트 문자열로 디코딩하거나 바이트 문자열을 16진법으로 인코딩 하기
--------------------------------------------------------------------------------------
'''

'''
--------------------------------------------------------------------------------------
- 문자열을 16진수로 인코딩하거나 디코딩하려면 binascii 모듈 사용
--------------------------------------------------------------------------------------
'''

# 최초 바이트 문자열
s = b'hello'

# 16진법으로 인코딩
import binascii

h = binascii.b2a_hex(s)

print(h)

# base64 모듈에도 유사한 기능
import base64

h = base64.b16encode(s)

print(h)
print(base64.b16decode(h))

# 유니코드 사용
h = base64.b16encode(s)
print(h.decode('ascii'))

'''
--------------------------------------------------------------------------------------
- base64.b16decode() 와 base64.b16encode() 함수는 대문자에만 동작

- binascii 는 대소문자를 가리지 않는다
--------------------------------------------------------------------------------------
'''





'''
--------------------------------------------------------------------------------------
Chapter7 함수
--------------------------------------------------------------------------------------
'''

'''
--------------------------------------------------------------------------------------
7.1 매개변수 개수에 구애 받지 않는 함수 작성

문제 : 입력 매개변수 개수에 제한이 없는 함수 작성하기
--------------------------------------------------------------------------------------
'''

'''
--------------------------------------------------------------------------------------
- 위치 매게변수의 개수에 제한이 없는 함수를 작성하려면 * 인자 사용
--------------------------------------------------------------------------------------
'''

def avg(first, *rest):
    return (first + sum(rest)) / (1+len(rest))

print(avg(1, 2))
print(avg(1, 2, 3, 4))

'''
--------------------------------------------------------------------------------------
- 키워드 매개변수 수에 제한이 없는 함수를 작성하려면 ** 인자 사용
--------------------------------------------------------------------------------------
'''

import html

def make_element(name, value, **attrs):
    keyvals = [' %s="%s"' %item for item in attrs.items()]
    attrs_str = ''.join(keyvals)
    element = '<{name}{attrs}>{value}</{name}>'.format(
                name=name,
                attrs=attrs_str,
                value=html.escape(value))
    return element

# 예제
# '<item size="large" quantity="6">Albatross</iteml>' 생성
print(make_element('item', 'Albatross', size='large', quantity=6))

# '<p>&lt;spam&gt;</p>' 생성
print(make_element('p', '<spam>'))

'''
--------------------------------------------------------------------------------------
- 위치 매개변수와 키워드 매개변수를 동시에 받는 함수 작성하려면 * 와 ** 인자 함께 사용
--------------------------------------------------------------------------------------
'''

def anyargs(*args, **kwargs):
    print(args)
    print(kwargs)

'''
=> 이 함수에서 모든 위치 매개변수는 튜플 args
             모든 키워드 매개변수는 딕셔너리 kwargs 에 들어간다
--------------------------------------------------------------------------------------
'''





'''
--------------------------------------------------------------------------------------
7.2 키워드 매개변수만 받는 함수 작성

문제 : 키워드로 지정한 특정 매개변수만 받는 함수 필요
--------------------------------------------------------------------------------------
'''

'''
--------------------------------------------------------------------------------------
- 키워드 매개변수를 * 뒤에 넣거나 이름 없이 * 만 사용하면 간단히 구현
--------------------------------------------------------------------------------------
'''

def recv(maxsize, *, block):
    'Receives a message'
    pass

print(recv(1024, True))         # TypeError
print(recv(1024, block=True))

'''
--------------------------------------------------------------------------------------
- 숫자가 다른 위치 매개변수를 받는 함수에 키워드 매개변수를 명시할 때 사용
--------------------------------------------------------------------------------------
'''

def mininum(*values, clip=None):
    m = min(values)
    if clip is not None:
        m = clip if clip > m else m
    return m

print(mininum(1, 5, 2, -5, 10))
print(mininum(1, 5, 2, -5, 10, clip=0))

'''
--------------------------------------------------------------------------------------
- 키워드로만 넣을 수 있는(keyword-only) 인자는 추가적 함수 인자를 명시할 때 코드의 가독성을 높이는 좋은 수단

- 키워드로만 넣을 수 있는 인자는 **kwargs 와 관련된 것에 사용자가 도움을 요청하면 도움말 화면에 나타낸다
  help(recv)
  Help on function recv in module __main__:
--------------------------------------------------------------------------------------
'''





'''
--------------------------------------------------------------------------------------
7.3 함수 인자에 메터데이터 넣기

문제 : 함수를 작성, 이때 인자에 정보를 추가해서 다른 사람이 함수를 어떻게 사용해야 하는지 알기
--------------------------------------------------------------------------------------
'''

'''
--------------------------------------------------------------------------------------
- 함수 인자 주석(function argument annotation) 으로 프로그래머에게 이 함수 사용 정보를 줄 수 있다
--------------------------------------------------------------------------------------
'''

def add(x:int, y:int) -> int:
    return x + y

help(add)

'''
--------------------------------------------------------------------------------------
- 함수 주석은 함수의 __annotations__ 속성에 저장
--------------------------------------------------------------------------------------
'''

def add(x:int, y:int) -> int:
    return x + y

print(add.__annotations__)





'''
--------------------------------------------------------------------------------------
7.4 함수에서 여러 값을 반환

문제 : 함수에서 값을 여러 개 반환하기
--------------------------------------------------------------------------------------
'''

'''
--------------------------------------------------------------------------------------
- 함수에서 값을 여러 개 반환하고 싶다면 간단히 튜플 사용
--------------------------------------------------------------------------------------
'''

def myfun():
    return 1, 2,3

a, b, c = myfun()

print(a)
print(b)
print(c)

a = (1, 2)  # 괄호 사용
print(a)

b = 1, 2    # 괄호 미사용
print(b)

x = myfun()
print(x)

'''
=> myfun() 이 값을 여러 개 반환하는 것처럼 보이지만, 사실은 튜플 하나를 반환한 것
   실제로 튜플을 생성하는 것은 쉼표지 괄호가 아니다 
   
=> 튜플을 반환하는 함수를 호출할 때, 결과 값을 여려 개의 변수에 넣는 것이 일반적
--------------------------------------------------------------------------------------
'''





'''
--------------------------------------------------------------------------------------
7.5 기본 인자를 사용하는 함수 정의

문제 : 함수나 메소드를 정의할 때 하나 혹은 그 이상 인자에 기본 값을 넣어 선택적으로 사용하기
--------------------------------------------------------------------------------------
'''

'''
--------------------------------------------------------------------------------------
- 함수 정의부에 값을 할당하고 가장 뒤에 이를 위치시키기
--------------------------------------------------------------------------------------
'''

def spam(a, b=42):
    print(a, b)

spam(1)     # a=1, b=42
spam(1, 2)  # a=1, b=2

'''
--------------------------------------------------------------------------------------
- 기본 값이 리스트, 세트, 딕셔너리 등 수정 가능한 컨테이너여야 한다면 None 사용
--------------------------------------------------------------------------------------
'''

# 기본 값으로 리스트 사용
def spam(a, b=None):
    if b is None:
        b = []

'''
--------------------------------------------------------------------------------------
- 기본 값을 제공하는 대신 함수가 받은 값이 특정 값인지 아닌지 확인하는 예제
--------------------------------------------------------------------------------------
'''

_no_value = object()

def spam(a, b=_no_value):
    if b is _no_value:
        print('No b value supplied')

spam(1)
spam(1, 2)
spam(1, None)

'''
--------------------------------------------------------------------------------------
- 할당하는 기본 값은 함수를 정의할 때 한 번만 정해지고 그 이후에는 변하지 않는다 
--------------------------------------------------------------------------------------
'''

x = 42

def spam(a, b=x):
    print(a, b)

spam(1)

x = 23
spam(1)     # x = 23 효과 없음!

'''
=> 변수 x(기본값으로 사용)의 값을 바꾸어도 그 이후에는 기본 값이 변하지 않는다
   기본 값은 함수를 정의할 때 정해지기 때문이다
--------------------------------------------------------------------------------------
'''

'''
--------------------------------------------------------------------------------------
- 기본 값으로 사용하는 값은 None, True, False, 숫자, 문자열 같이 항상 같이 변하지 않는 객체 사용

- 기본 깂이 함수를 벗어나서 수정되는 순간 많은 문제가 발생
  값이 변하면 기본 값이 변하게 되고 추후 함수 호출에 영향을 준다
--------------------------------------------------------------------------------------
'''

def spam(a, b=[]):
    print(b)
    return b

x = spam(1)
print(x)

x.append(99)
x.append('Yow!')

spam(1)         # 수정된 리스트가 반환!!

'''
=> 이런 부작용을 피하려면 기본 값으로 None 을 할당하고 함수 내부에서 이를 확인

=> None을 확인할 때 is 연산자를 사용하는 것이 매우 중요!
--------------------------------------------------------------------------------------
'''

def spam(a, b=None):
    if not b:
        b = []

'''
=> 여기서 문제는 None 이 False로 평가되지만, 그 외에 다른 객체(길이가 0인 문자열, 리스트, 튜플, 딕셔너리)도 False로 평가된다
--------------------------------------------------------------------------------------
'''

'''
--------------------------------------------------------------------------------------
- 사용자가 인자를 넣었는지 확인할 때 기본값으로 None, 0 또는 False를 사용할 수 없다!
  사용자가 바로 이 값을 인자로 넣을 수도 있기 때문이다
  
- 위 문제를 해결하기 위해서 앞의 예제(_no_value 변수)에 나온 것처럼 object의 유일한 인스턴스 만들기
--------------------------------------------------------------------------------------
'''





'''
--------------------------------------------------------------------------------------
7.6 이름 없는 함수와 인라인 함수 정의

문제 : sort() 등에 사용할 짧은 콜백함수를 만들어야 하는데 , 한 줄 짜리 함수를 만들면서 def 구문까지 사용하지 않고
      그 대신 인 라인(in line) 이라 불리는 짧은 함수 만들기
--------------------------------------------------------------------------------------
'''

