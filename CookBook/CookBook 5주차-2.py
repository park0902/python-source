'''
--------------------------------------------------------------------------------------
8.6 관리 속성 만들기

문제 : 인스턴스 속성을 얻거나 설정할 때 추가적인 처리(타입 체크, 검증 등) 하기
--------------------------------------------------------------------------------------
'''

'''
--------------------------------------------------------------------------------------
- 속성에 대한 접근을 조절하고 싶으면 프로퍼티(property) 로 정의
  속성에 간단한 타입 체크를 추가하는 프로퍼티 정의 예제
--------------------------------------------------------------------------------------
'''

class Person:
    def __init__(self, first_name):
        self.first_name = first_name

    # 게터 함수
    @property
    def first_name(self):
        return self._first_name

    # 세터 함수
    @first_name.setter
    def first_name(self, value):
        if not isinstance(value, str):
            raise TypeError('Expected a string')
        self._first_name = value

    # 딜리티 함수(옵션)
    @first_name.deleter
    def first_name(self):
        raise AttributeError("Can't delete attribute")

a = Person('Guido')

print(a.first_name)

'''
=> 메소드가 세 개 있는데, 모두 같은 이름을 가져야 한다

=> 첫번째 메소드는 게터 함수로 first_name 을 프로퍼티로 만든다
   다른 두 메소드는 추가적으로 세터와 딜리티 함수를 first_name 프로퍼티에 추가한다

=> @first_name.setter 와 @first_name.delete 데코레이터는 @property 를 사용해서 first_name 을 
   만들어 놓지 않으면 정의되지 않는 점이 중요!!
   
=> 프로퍼티를 구현할 때, 기반 데이터가 있다면 여전히 어딘가에 저장해야 한다
   따라서 게터, 세터 메소드에서 _first_name 속성을 직접 다루는 것을 볼 수 있는데, 여기에 실제 데이터가 들어간다
--------------------------------------------------------------------------------------
'''

'''
--------------------------------------------------------------------------------------
- 이미 존재하는 get 과 set 메소드로 프로퍼티 정의
--------------------------------------------------------------------------------------
'''

class Person:
    def __init__(self, first_name):
        self.set_first_name(first_name)

    # 게터 함수
    def get_first_name(self):
        return self._first_name

    # 세터 함수
    def set_first_name(self, value):
        if not isinstance(value, str):
            raise TypeError('Expected a string')
        self._first_name = value

    # 딜리티 함수(옵션)
    def del_first_name(self):
        raise AttributeError("Can't delete attribute")

# 기존 게터/세터 메소드로 프로퍼티 만들기
name = property(Person.get_first_name, Person.set_first_name, Person.del_first_name)

'''
--------------------------------------------------------------------------------------
- 일반적으로 fget 이나 fset을 직접 호출하지는 않고, 프로퍼티에 접근할 때 자동으로 실행된다

- 프로퍼티는 속성에 추가적인 처리가 필요할 때만 사용해야 한다
--------------------------------------------------------------------------------------
'''

class Person:
    def __init__(self, first_name):
        self.first_name = first_name

    @property
    def first_name(self):
        return self._first_name

    @first_name.setter
    def first_name(self, value):
        self._first_name = value

'''
--------------------------------------------------------------------------------------
- 프로퍼티는 계산한 속성을 정의할 때 사용하기도 한다
  이런 속성은 실제로 저장하지는 않지만 필요에 따라 계산을 한다
--------------------------------------------------------------------------------------
'''

import math

class Circle:
    def __init__(self, radius):
        self.radius = radius

    @property
    def area(self):
        return math.pi * self.radius ** 2

    @property
    def perimeter(self):
        return 2 * math.pi * self.radius

c = Circle(4.0)

print(c.radius)
print(c.area)
print(c.perimeter)

'''
--------------------------------------------------------------------------------------
- 프로퍼티 정의를 반복적으로 사용하는 파이썬 코드를 작성하지 않도록 주의!
--------------------------------------------------------------------------------------
'''

class Person:
    def __init__(self, first_name, last_name):
        self.first_name = first_name
        self.last_name = last_name

    @property
    def first_name(self):
        return self._first_name

    @first_name.setter
    def first_name(self, value):
        if not isinstance(value, str):
            raise TypeError('Expected a string')
        self._first_name = value

    # 이름이 다른 프로퍼티 코드의 반복 (좋지 않다!!!!)
    @property
    def last_name(self):
        return self._last_name

    @last_name.setter
    def last_name(self, value):
        if not isinstance(value, str):
            raise TypeError('Expected a string')
        self._last_name = value





'''
--------------------------------------------------------------------------------------
8.7 부모 클래스의 메소드 호출

문제 : 오버라이드된 서브클래스 메소드가 아닌 부모 클래스에 있는 메소드를 호출하기
--------------------------------------------------------------------------------------
'''

'''
--------------------------------------------------------------------------------------
- 부모 (혹은 슈퍼클래스)의 메소드를 호출하려면 super() 함수 사용
--------------------------------------------------------------------------------------
'''

class A:
    def spam(self):
        print('A.spam')

class B(A):
    def spam(self):
        print('B.spam')
        super().spam()      # 부모의 spam() 호출


# super() 는 일반적으로 __init__() 메소드에서 부모를 제대로 초기화하기 위해 사용
class A:
    def __int__(self):
        self.x = 0

class B:
    def __init__(self):
        super().__init__()
        self.y = 1

# 파이썬의 특별 메소드를 오버라이드한 코드에서 super() 를 사용하기도 한다
class Proxy:
    def __int__(self, obj):
        self.obj = obj

    # 내부 obj를 위해 델리게이트(delgate) 속성 찾기
    def __getattr__(self, name):
        return getattr(self.obj, name)

    # 델리게이트(delgate) 속성 할당
    def __setattr__(self, name, value):
        if name.startswith('-'):
            super().__setattr__(name, value)    # 원본 __setattr__ 호출

        else:
            setattr(self.obj, name, value)

'''
=> __setattr__() 구현에 이름 확인이 들어있다

=> 만약 이름이 밑줄로 시작하면 super()를 사용해서 __setattr__() 의 원래의 구현을 호출
   그렇지 않다면 내부 객체인 self._obj 를 부른다
--------------------------------------------------------------------------------------
'''

'''
--------------------------------------------------------------------------------------
- 부모 클래스 메소드를 직접 호출하기 위한 예제
--------------------------------------------------------------------------------------
'''

class Base:
    def __init__(self):
        print('Base.__init__')

class A(Base):
    def __int__(self):
        Base.__init__(self)
        print('A.__init__')

'''
--------------------------------------------------------------------------------------
- 다중 상속과 같은 상황에서 문제 발생 예제
--------------------------------------------------------------------------------------
'''

class Base:
    def __int__(self):
        print('Base.__init__')

class A(Base):
    def __int__(self):
        Base.__init__(self)
        print('A.__init__')

class B(Base):
    def __int__(self):
        Base.__init__(self)
        print('B.__init__')

class C(A, B):
    def __init__(self):
        A.__init__(self)
        B.__init__(self)
        print('C.__init__')

c = C()     # Base.__init__() 메소드가 두 번 호출된다

'''
--------------------------------------------------------------------------------------
- super() 를 사용하여 코드 수정 예제
--------------------------------------------------------------------------------------
'''


class Base:
    def __int__(self):
        print('Base.__init__')


class A(Base):
    def __int__(self):
        super().__init__()
        print('A.__init__')


class B(Base):
    def __int__(self):
        super().__init__()
        print('B.__init__')


class C(A, B):
    def __init__(self):
        super().__init__()
        print('C.__init__')

c = C()     # 여기서 super() 를 한 번만 호출한다

'''
--------------------------------------------------------------------------------------
- MRO(Method Resolution Order 메소드 처리 순서) 리스트 자체를 실제로 결정할 때는 C3 선형화 기술 사용

- 너무 계산이 복잡해지지 않도록 부모 클래스의 MRO 를 세 가지 제약 조건 하에서 합병 정렬(merge sort) 한다

    자식 클래스를 부모보다 먼저 확인한다
    부모 클래스가 둘 이상이면 리스팅 순서대로 확인한다
    유효한 후보가 두 가지 있으면, 첫번째 부모 클래스부터 실행한다
--------------------------------------------------------------------------------------
'''