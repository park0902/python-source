# class Gun():
#     def __init__(self):
#         self.bullet = 0
#
#     def charge(self, num):  # 충전하는 기능
#         self.bullet = num
#
#     def shoot(self, num):     # 쏘는 기능
#         for i in range(num):
#             if self.bullet > 0:
#                 print('탕')
#                 self.bullet -= 1
#             elif self.bullet == 0:
#                 print('총알이 없습니다')
#                 break
#
#     def print(self):        # 출력하는 기능
#         print('{} 발 남았습니다'.format(self.bullet))
#
# gun = Gun()
#
# gun.charge(10)
# gun.shoot(3)



# class Student(object):
#     def __init__(self,name,year,class_num,student_id):
#         self.name = name
#         self.year = year
#         self.clss_num = class_num
#         self.student_id = student_id
#
#     def introduce_myself(self):
#         return '{}, {}학년 {}반 {}번'.format(self.name, self.year,
#                                             self.clss_num,self.student_id)
#
# student_1 = Student('김인호',2,3,35)
#
# print(Student.introduce_myself(student_1))


# class OnlyAdmin(object):
#     def __init__(self, func):  # 대문자로 변환해주며 함수를 실행해주는 클래스
#         self.func = func
#
#     def __call__(self, *args, **kwargs):
#         name = kwargs.get('name').upper()
#         return self.func(name)
#
# @OnlyAdmin
# def find_job(name):
#
#     import pandas as pd
#     emp = pd.read_csv('d:\data\emp.csv')
#
#     job = emp['job'][emp['ename'] == name].values[0]
#     return '당신의 직업은 {} 입니다'.format(job)
#
#
# print(find_job(name='scott'))

from abc import ABCMeta, abstractclassmethod  # 파이썬을 추상 클래스 제공
                                              # 하지 않아서 import
class Animal(object):
    __metaclass__ = ABCMeta     # 추상 클래스 선언

    @abstractclassmethod
    def bark(self):
        pass        # 비어있는 메소드, 상속받는 자식들이 반드시
                    # 구현해야되는 중요한 메소드

class Cat(Animal):
    def __init__(self):
        self.sound = 23.5

    def bark(self):
        return self.sound

class Dog(Animal):
    def __init__(self):
        self.sound = 12.1

    def bark(self):
        return self.sound

animal = Animal()
cat = Cat()
dog = Dog()

print(cat.bark())
print(dog.bark())


