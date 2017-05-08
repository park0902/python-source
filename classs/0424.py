# def print_inform(name, team='머신러닝팀', position='팀원'):
#     print('이름 = {0}'.format(name))
#     print('소속팀 = {0}'.format(team))
#     print('직위 = {0}'.format(position))
#
# print_inform(name='박상범', team = '파이썬')


# States as integer : manual coding
# EMPTY = 0
# PLAYER_X = 1
# PLAYER_O = 2
# DRAW = 3
#
# BOARD_FORMAT = """----------------------------
# | {0} | {1} | {2} |
# |--------------------------|
# | {3} | {4} | {5} |
# |--------------------------|
# | {6} | {7} | {8} |
# ----------------------------"""
#
# NAMES = [' ', 'X', 'O']
#
# def printboard(state):
#     """ Print the board from the internal state."""
#     cells = []
#     for i in range(3):
#         for j in range(3):
#             cells.append(NAMES[state[i][j]].center(6))
#             print(i,j)
#             print(state[i][j])
#     print(cells)
#     print(*cells)
#     print(BOARD_FORMAT.format(*cells))
#
# printboard([[1,2,0],[0,1,0],[0,0,0]])
# print(BOARD_FORMAT.format('a','b','c','d','e','f','g','h','g'))
# print(BOARD_FORMAT.format('x','o','x','o','x',' ',' ',' ',' '))
# print(BOARD_FORMAT.format('x'.center(6),'o','x','o','x',' ',' ',' ',' '))

# def scope_test():
#     # 글로벌 변수로 선언
#     a = 1  # 함수 내에서 사용하는 변수(로컬 변수)
#     print('a : {0}'.format(a))
#
#
# a = 0
#
# scope_test()
#
# print('a : {0}'.format(a))


# def find_gcd(num1,num2,num3):
#     res = num1%num2%num3
#     if res != 0:
#         res = num1%num2%num3
#         return find_gcd(num2,num3,res)
#     else:
#         return num3
#
# print('최대공약수 ',find_gcd(144,60,48))
#
#
#
# def find_gcd(num1,num2,num3):
#     res = num1%num2%num3
#     if res == 0:
#         return num3
#     else:
#         res = num1%num2%num3
#         return find_gcd(num2,num3,res)
#
# print('최대공약수 ',find_gcd(144,60,48))


# def find_gcd(num1,num2):
#     c = num1%num2
#     if num2%c == 0:
#         return c
#     else:
#         find_gcd(num2,c)
#
# print('최대공약수 ',find_gcd(36,16))