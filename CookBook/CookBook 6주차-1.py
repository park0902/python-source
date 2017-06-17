'''
--------------------------------------------------------------------------------------
8.12 인터페이스, 추상 베이스 클래스 정의

문제 : 인터페이스나 추상 베이스 클래스 역할을 하는 클래스를 정의하고 이 클래스는 타입 확인을 하고 특정 메소드가
      서브 클래스에 구현되었는지 보장하기
--------------------------------------------------------------------------------------
'''

'''
--------------------------------------------------------------------------------------
- 추상 베이스 클래스를 정의하려면 abc 모듈 사용
--------------------------------------------------------------------------------------
'''

from abc import ABCMeta, abstractclassmethod

class IStream(metaclass=ABCMeta):
    @abstractclassmethod
    def read(self, maxbytes=-1):
        pass
    @abstractclassmethod
    def write(self, data):
        pass

'''
=> 추상 베이스 클래스의 주요 기능은 직접 인스턴스화 할 수 없다!!!

=> 추상 베이스 클래승는 요구한 메소드를 구현하는 다른 클래스의 베이스 클래스로 사용해야 한다!!

=> 추상 클래스는 특정 프로그래밍 인터페이스를 강요하고 싶을 때 주로 사용!!
--------------------------------------------------------------------------------------
'''

'''
--------------------------------------------------------------------------------------
- 인터페이스를 명시적으로 확인하는 예제
--------------------------------------------------------------------------------------
'''

def serialize(obj, stream):
    if not isinstance(stream, IStream):
        raise TypeError('Expeected an IStream')
    ...

'''
--------------------------------------------------------------------------------------
- ABC는 다른 클래스가 특정 인터페이스를 구현하는 예제
--------------------------------------------------------------------------------------
'''

import io

# 내장 I/O 클래스를 우리의 인터페이스를 지원하도록 등록
IStream.register(io.IOBase)

# 일반 파일을 열고 타입 확인
f = open('e:\data\somefile.txt')

print(isinstance(f, IStream))

'''
--------------------------------------------------------------------------------------
- @abstractmethod 를 스태틱 메소드(static method), 클래스 메소드, 프로퍼티에도 적용할 수 도 있다
--------------------------------------------------------------------------------------
'''

class A(metaclass=ABCMeta):
    @property
    @abstractclassmethod
    def name(self):
        pass

    @name.setter
    @abstractclassmethod
    def name(self, value):
        pass

    @classmethod
    @abstractclassmethod
    def method(cls):
        pass

    @staticmethod
    @abstractclassmethod
    def method2():
        pass

'''
--------------------------------------------------------------------------------------
- 추상 베이스 클래스를 더 일반적인 타입 확인에 사용 가능
--------------------------------------------------------------------------------------
'''

import collections

x = ''

# x 가 시퀀스인지 확인
if isinstance(x, collections.Sequence):
    ...

# x 가 순환 가능한지 확인
if isinstance(x, collections.Iterable):
    ...

# x 가 크기가 있는지 확인
if isinstance(x, collections.Sized):
    ...

# x 가 매핑인지 확인
if isinstance(x, collections.Mapping):
    ...





'''
--------------------------------------------------------------------------------------
8.13 관리 속성 만들기

문제 : 여러 종류의 자료 구조를 정의하고 싶다. 이때 특정 값에 제약을 걸어 원하는 속성이 할당되도록 하기
--------------------------------------------------------------------------------------
'''

'''
--------------------------------------------------------------------------------------
- 디스크립터로 시스템 타입과 값 확인 프레임워크를 구현 예제
--------------------------------------------------------------------------------------
'''

# 베이스 클래스, 디스크립터로 값을 설정한다
class Descriptor:
    def __init__(self, name=None, **opts):
        self.name = name
        for key, value in opts.items():
            setattr(self, key, value)

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value

# 타입을 강제하기 위한 디스크립터
class Typed(Descriptor):
    expected_type = type(None)

    def __set__(self, instance, value):
        if not isinstance(value, self.expected_type):
            raise TypeError('expected' + str(self.expected_type))
        super().__set__(instance, value)

# 값을 강제하기 위한 디스크립터
class Unsigned(Descriptor):
    def __set__(self, instance, value):
        if value < 0:
            raise ValueError('Expected >= 0')
        super().__set__(instance, value)

class MaxSized(Descriptor):
    def __int__(self, name=None, **opts):
        if 'size' not in opts:
            raise TypeError('missing size option')
        super().__init__(name, **opts)

    def __set__(self, instance, value):
        if len(value) >= self.size:
            raise ValueError('size must be' + str(self.size))
        super().__set__(instance, value)

'''
--------------------------------------------------------------------------------------
- 서로 다른 데이터를 구현하는 예제
--------------------------------------------------------------------------------------
'''

class Integer(Typed):
    expected_type = int

class UnisgnedInteger(Integer, Unsigned):
    pass

class Float(Typed):
    expected_type = float

class UnsignedFloat(Float, Unsigned):
    pass

class String(Typed):
    expected_type = str

class SizedString(String, MaxSized):
    pass

'''
--------------------------------------------------------------------------------------
- 타입 객체를 사용해서 다음과 같은 클래스를 정의
--------------------------------------------------------------------------------------
'''

class Stock:
    # 제약 명시
    name = SizedString('name', size=8)
    shares = UnisgnedInteger('shares')
    price = UnsignedFloat('price')

    def __init__(self, name, shares, price):
        self.name = name
        self.shares = shares
        self.price = price

s = Stock('ACME', 50, 91.1)
s.shares = 75
s.shares = -10          # 에러
s.price = 'a lot'       # 에러
s.name = 'ABRACADABRA'  # 에러

print(s.name)

'''
--------------------------------------------------------------------------------------
- 클래스 데코레이터 사용하는 예제
--------------------------------------------------------------------------------------
'''

# 제약을 위한 클래스 데코레이터
def check_attributes(**kwargs):
    def decorate(cls):
        for key, value in kwargs.items():
            if isinstance(value, Descriptor):
                value.name = key
                setattr(cls, key, value)
            else:
                setattr(cls, key, value(key))
        return cls
    return decorate

# 예제
@check_attributes(name=SizedString(size=8),
                  shares=UnisgnedInteger,
                  price=UnsignedFloat)

class Stock:
    def __init__(self, name, shares, price):
        self.name = name
        self.shares = shares
        self.price = price

# 확인을 위한 메타클래스
class checkedmeta(type):
    def __new__(cls, clasname, bases, methods):
        # 디스크립터에 속성 이름 붙이기
        for key, value in methods.items():
            if isinstance(value, Descriptor):
                value.name = key
        return type.__new__(cls, clasname, bases, methods)

# 예제
class Stock(metaclass=checkedmeta):
    name = SizedString(size=8)
    shares = UnisgnedInteger()
    price = UnsignedFloat()
    def __init__(self, name, shares, price):
        self.name = name
        self.shares = shares
        self.price = price

'''
--------------------------------------------------------------------------------------
- 클래스 데코레이터나 메타클래스를 사용하면 사용자의 스펙을 단순화할 때 유용
--------------------------------------------------------------------------------------
'''

# 일반
class Point:
    x = Integer('x')
    y = Integer('y')

# 메타클래스
class Point(metaclass=checkedmeta):
    x = Integer()
    y = Integer()

'''
--------------------------------------------------------------------------------------
- 클래스 데코레이터의 방식은 믹스인(mixin) 클래스, 다중 상속, 복잡한 super() 대신 사용할 수 있다
--------------------------------------------------------------------------------------
'''

# 베이스 클래스, 값을 설정할 때 디스크립터를 사용
class Descriptor:
    def __init__(self, name=None, **opts):
        self.name = name
        for key, value in opts.items():
            setattr(self, key, value)

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value

# 타입 확인에 데코레이터 사용
def Typed(expected_type, cls=None):
    if cls is None:
        return lambda cls: Typed(expected_type, cls)

    super_set = cls.__set__
    def __set__(self, instance, value):
        if not isinstance(value, expected_type):
            raise TypeError('expected' + str(expected_type))
        super_set(self, instance, value)
    cls.__set__ = __set__
    return cls

# 언사인드(unsigned) 값에 데코레이터 사용
def Unsigned(cls):
    super_set = cls.__set__
    def __set__(self, instance, value):
        if value < 0:
            raise ValueError('Expected >= 0')
        super_set(self, instance, value)
    cls.__set__ = __set__
    return cls

# 크기 있는 값에 데코레이터 사용
def MaxSized(cls):
    super_init = cls.__init__
    def __init__(self, name=None, **opts):
        if 'size' not in opts:
            raise TypeError('missing size option')
        super_init = __init__
    cls.__init__ = __init__

    super_set = cls.__set__
    def __set__(self, instane, value):
        if len(value) >= self.size:
            raise ValueError('size must be < ' + str(self.size))
        super_set(self, instane, value)
    cls.__set__ = __set__
    return cls

# 특별 디스크립터
@Typed(int)
class Integer(Descriptor):
    pass

@Unsigned
class UnsignedInteger(Integer):
    pass

@Typed(float)
class Float(Descriptor):
    pass

@Unsigned
class UnsignedFloat(Float):
    pass

@Typed(str)
class String(Descriptor):
    pass

@MaxSized
class SizedString(String):
    pass





'''
--------------------------------------------------------------------------------------
8.14 커스텀 컨테이너 구현

문제 : 리스트나 딕셔너리와 같은 내장 컨테이너와 비슷하게 동작하는 커스텀 클래스를 구현하는데
      하지만 이때 정확히 어떤 메소드를 구현해야 할지 확신이 없다
--------------------------------------------------------------------------------------
'''

'''
--------------------------------------------------------------------------------------
- collections 라이브러리에 이 목적으로 사용하기 적절한 추상 베이스 클래스가 많이 정의되어 있다
--------------------------------------------------------------------------------------
'''

import collections

class A(collections.Iterable):
    pass

a = A()     # 에러

'''
=> collections.Iterable 을 상속 받으면 필요한 모든 특별 메소드를 구현하도록 보장해 준다
 
=> 메소드 구현을 잊으면 인스턴스화 과정에서 에러 발생!

=> 에러를 고치려면 클래스가 필요로 하는 __iter__() 매소드 구현
--------------------------------------------------------------------------------------
'''

'''
--------------------------------------------------------------------------------------
- 필요한 메소드를 모두 구현해서 아이템을 정렬된 상태로 저장하는 시퀀스 예제
--------------------------------------------------------------------------------------
'''

import collections
import bisect

class SortedItems(collections.Sequence):
    def __init__(self, initial=None):
        self._items = sorted(initial) if initial is None else []

    # 필요한 시퀀스 메소드
    def __getitem__(self, index):
        return self._items[index]

    def __len__(self):
        return len(self._items)

    # 올바른 장소에 아이템을 추가하기 위한 메소드
    def add(self, item):
        bisect.insort(self._items, item)

items = SortedItems([5, 1, 3])
items.add(2)

print(list(items))
print(items[0])
print(items[-1])
print(items[1:4])
print(3 in items)
print(len(items))

for n in items:
    print(n)

'''
=> SortedItems의 인스턴스는 보통의 시퀀스와 동일한 동작을 하고 인덱싱, 순환, 
   len(), in 연산자, 자르기 등 일반적인 연산 모두 지원
   
=> bisect 모듈은 아이템을 정렬한 상태로 리스트에 보관할때 매우 편리
   bisect.insort() 는 아이템을 리스트에 넣고 리스트가 순서를 유지하도록 만든다
--------------------------------------------------------------------------------------
'''

'''
--------------------------------------------------------------------------------------
- 커스텀 컨테이너에 타입 확인 예제
--------------------------------------------------------------------------------------
'''

items = SortedItems()

import collections

print(isinstance(items, collections.Iterable))
print(isinstance(items, collections.Sequence))
print(isinstance(items, collections.Container))
print(isinstance(items, collections.Sized))
print(isinstance(items, collections.Mapping))

'''
--------------------------------------------------------------------------------------
- collections.MutableSequence 에서 상속 받는 클래스 예제
--------------------------------------------------------------------------------------
'''

import collections

class Items(collections.MutableSequence):
    def __init__(self, initial=None):
        self._items = list(initial) if initial is None else []

    # 필요한 시퀀스 메소드
    def __getitem__(self, index):
        print('Getting : ', index)
        return self._items[index]

    def __setitem__(self, index, value):
        print('Setting : ', index, value)
        self._items[index] = value

    def __delitem__(self, index):
        print('Deleting : ', index)
        del self._items[index]

    def insert(self, index, value):
        print('Inserting : ', index, value)

    def __len__(self):
        print('Len')
        return len(self._items)

a = Items([1, 2, 3])

print(len(a))
print(a.append((4)))
print(a.append((2)))
print(a.count(2))
print(a.remove(3))
















