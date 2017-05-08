# class YourClass:
#     pass
#
# class MyClass:
#     def __init__(self):
#         self.message = 'Hello'
#         self.__private = 'private'
#
#     def some_method(self):
#         print(self.message)
#         print(self.__private)
#
# obj = MyClass()
#
# obj.some_method()   # 메소드를 실행했기 때문에 출력
# print(obj.message)  # message 라는 변수의 내용 출력
# print(obj.__pivate) # 프라이빗 멤버 접근 불가여서 에러


# class Father:
#     def __init__(self):
#         print("Hello~~")
#         self.message = 'Good Morning'
#
# class Child(Father):
#     def __init__(self):
#         #Father.__init__(self)   # 이 부분을 해야 child.message 출력
#         super().__init__()
#         print('Hello~~ I am Child')
#
# father = Father()
# child = Child()
#
# print(father.message)
# print(child.message)




#
# class Father1:
#     def func(self):
#         print('지식')
#
# class Father2:
#     def func(self):
#         print('지혜')
#
# class Child(Father1, Father2):
#     def childfunc(self):
#         Father1.func(self)
#         Father2.func(self)
#
# objectchild = Child()
# objectchild.func()



# class GrandFather:
#     def __init__(self):
#         print('튼튼한 두 팔')
#
# class Father1(GrandFather):
#     def __init__(self):
#         super().__init__()
#         #GrandFather.__init__(self)
#         print('지식')
#
# class Father2(GrandFather):
#     def __init__(self):
#         super().__init__()
#         #GrandFather.__init__(self)
#         print('지혜')
#
# class GrandChild(Father1, Father2):
#     def __init__(self):
#         super().__init__()
#         #Father1.__init__(self)
#         #Father2.__init__(self)
#         print('자기 만족도가 높은 삶')
#
# grandchild = GrandChild()

# class GrandFater:
#     def __init__(self):
#         print('튼튼한 두팔')
#
# class Father2(GrandFater):
#     def __init__(self):
#         super().__init__()
#         print('지혜')
#
# father2 = Father2()


# def find_job(name):
#
#     import pandas as pd
#
#     emp = pd.DataFrame.from_csv("d:/data/emp.csv")
#
#     job = emp[['job']][emp['ename'] == name].values[0][0]
#
#     return job
#
#
# def lowercase(func):
#     def wrapper(name):
#         result = func(name)
#
#         return result.lower()
#
#     return wrapper
#
#
# new_find_job = lowercase(find_job)
#
# print(new_find_job('SCOTT'))


#
# class Greet(object):
#     current_user = None         # current_user 라는 변수인 속성 선언
#     def set_name(self,name):    # name에 Admin 문자가 들어오면
#         if name == 'Admin':     # current_user에 Admin 문자를 담고
#             self.current_user = name
#         else:                   # Admin 이 아니라면 권한이 없다는 에러를
#             raise Exception('권한이 없습니다')
#             # 발생시키는 함수
#     def get_greeting(self,name):    # name 이라는 매개변수에 Admin 이
#         if name == 'Admin':         # 입력이 됬다면
#             return 'Hello {}'.format(self.current_user)
#             # Hello 와 current_user를 리턴하는 함수
#
# greet = Greet()
#
# greet.set_name('scott')
# print(greet.get_greeting('scott'))


# def is_admin(user_name):
#     if user_name != 'admin':
#         raise Exception('권한이 없습니다!!!!')
#
# class Greet(object):
#     current_user = None
#     def set_name(self,name):
#         is_admin(name)
#         self.current_user = name
#
#     def get_greeting(self,name):
#             is_admin(name)
#             return 'Hello {}'.format(self.current_user)
#
#
# greet = Greet()
#
# greet.set_name('scott')
# print(greet.get_greeting('scott'))




# def is_king(user_name):
#     if user_name != 'KING':
#         raise  Exception('권한이 없습니다!!')
#
# class find_sal(object):
#     current_user = None
#     def set_name(self, name):
#         is_king(name)
#         self.current_user = name
#
#     def get_sal(self,name):
#         is_king(name)
#
#         import pandas as pd
#
#         emp = pd.DataFrame.from_csv("d:/data/emp.csv")
#
#         sal = emp[['sal']][emp['ename'] == name].values[0][0]
#
#         return sal
#
# find_sal = find_sal()
# find_sal.set_name('SCOTT')
# print(find_sal.get_sal('SCOTT'))


# def is_king(func):
#     def wrapper(*args, **kwargs):
#         if kwargs.get('name') != 'KING':
#             raise  Exception('권한이 없습니다!!')
#         return func(*args, **kwargs)
#     return wrapper
#
# class find_sal(object):
#     current_user = None
#     @is_king
#     def set_name(self, name):
#         self.current_user = name
#
#     @is_king
#     def get_sal(self,name):
#
#         import pandas as pd
#         emp = pd.DataFrame.from_csv("d:/data/emp.csv")
#         sal = emp[['sal']][emp['ename'] == name].values[0][0]
#
#         return sal
#
# find_sal = find_sal()
# find_sal.set_name(name='KING')
# print(find_sal.get_sal(name='KING'))