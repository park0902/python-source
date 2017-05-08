# class Car:
#     def __init__(self):
#         self.color = 0xFF000        # 차 색깔
#         self.wheel_size = 16        # 바퀴 크기
#         self.displacement = 2000    # 엔진 배기량
#
#     def foward(self):       # 전진 기능
#         pass
#
#     def backward(self):     # 후진 기능
#         pass
#
#     def turn_left(self):    # 좌회전 기능
#         pass
#
#     def turn_right(self):   # 우회전 기능
#         pass
#
# if __name__=='__main__':    # 메인모듈일때 아래 스크립트를 실행!
#
#     my_car = Car()          # 클래스를 가지고 인스턴스화 하는 코드
#                             # Car() 클래스로 my_car 라는 객체 생성
#
#     print('0x{:2X}'.format(my_car.color))   # my_car 에 대한 정보 출력
#     print(my_car.wheel_size)
#     print(my_car.displacement)
#
#     my_car.foward()         # my_car 의 메소드(기능) 호출
#     my_car.backward()
#     my_car.turn_right()
#     my_car.turn_left()



# class Calculator:   # 기능만 4가지 있는 클래스
#     @staticmethod   # 정적 메소드를 선언할때 사용해야하는 데코레이터
#     def plus(a, b):
#         return a + b
#
#     @staticmethod
#     def minus(a, b):
#         return a - b
#
#     @staticmethod
#     def multiply(a, b):
#         return a * b
#
#     @staticmethod
#     def divide(a, b):
#         return a / b
#
#
# if __name__ == '__main__':
#     print("{0} + {1} = {2}".format(7, 4, Calculator.plus(7, 4)))
#     print("{0} - {1} = {2}".format(7, 4, Calculator.minus(7, 4)))
#     print("{0} * {1} = {2}".format(7, 4, Calculator.multiply(7, 4)))
#     print("{0} / {1} = {2}".format(7, 4, Calculator.divide(7, 4)))
# #                                            클래스 . 함수
# #   정적 메소드를 호출하는 방법(클래스를 통해서 메소드 호출)