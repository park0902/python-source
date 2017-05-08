# class Computer(object):
#     def __init__(self, player):
#         self.player = player
#         self.values = {}
#         self.readCSV() # init 할때 value 에 값 채워넣을려고 함수를 실행함
#         self.verbose = True # 자릿수의 확률을 확인하기 위해 사용
#         print(self.values) # csv 파일의 데이터를 읽어와서 values 딕셔너리에 데이터 입력
#


# import random
#
# state = [[1,2,0], [0,0,0], [0,0,0]]
# EMPTY = 0
#
# def random2(state):
#     available = []
#     for i in range(3):
#         for j in range(3):
#             if state[i][j] == EMPTY:
#                 available.append((i ,j))
#
#     print(available)
#     return random.choice(available)
#
# print(random2(state))