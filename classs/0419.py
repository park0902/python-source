# import pandas as pd
# emp = pd.read_csv("D:\data\emp.csv")
#
# result = emp[['ename']][emp['empno'].isin(emp['mgr'])]
#
# print(result)

# import csv
#
# file = open("D:\data\emp_comm.csv", "r")
# emp_csv = csv.reader(file)
#
# for emp_list in emp_csv:
#     if emp_list[1][0] == 'S':
#         print(emp_list[1])


# import pandas as pd
# ttt = pd.read_csv("D:\data\mit_ttt2.csv")
#
# sub = ttt[(ttt['PLAYER'] == 1) &
#           (ttt['C1'] == 1) &
#           (ttt['C2'] == 2) &
#           (ttt['C3'] == 1) &
#           (ttt['C4'] == 2) &
#           (ttt['C7'] == 2) &
#           (ttt['C9'] == 1) &
#           (ttt['C5'] + ttt['C6'] + ttt['C8'] == 1)]
#
# res = sub.groupby(['C5', 'C6', 'C8'])['LEARNING_ORDER'].max()
#
# result = ttt[ttt['LEARNING_ORDER'].isin(res) & ttt['PLAYER'] == 1]
#
# print(result)

# import pandas as pd
# ttt = pd.read_csv("D:\data\mit_ttt2.csv")
#
# sub = ttt[(ttt['PLAYER'] == 1) &
#           (ttt['C1'] == 1) &
#           (ttt['C2'] == 2) &
#           (ttt['C3'] == 1) &
#           (ttt['C4'] == 2) &
#           (ttt['C7'] == 2) &
#           (ttt['C9'] == 1) &
#           (ttt['C5'] + ttt['C6'] + ttt['C8'] == 1)]
#
# res = sub.groupby(['C5', 'C6', 'C8'])['LEARNING_ORDER'].max()
#
# a = []
# for i in res:
#     a.append(i)
#
# print(a)
# print(type(a))
# result = ttt[ttt['LEARNING_ORDER'].isin(a) & ttt['PLAYER'] == 1]
#
# print(result)


# dic = {}
# dic['나는'] = ('I', 0)
# dic['소년'] = ('boy', 2)
# dic['이다'] = ('am', 1)
# dic['피자'] = ('Pizza', 2)
# dic['먹는다'] = ('eat', 1)
#
# result = ''
# input_kor = input('입력하세요.(나는 소년 이다 / 나는 피자를 먹는다) : ')
# input_list = input_kor.split(' ')
# for i in range(len(input_list)):
#     for j in input_list:
#         if dic[j][1] == i:
#             result = result + dic[j][0] + ' '
#
# print(result)


# import csv
#
# file = open("D:\data\smt_dic.csv", "r")
# smt_csv = csv.reader(file)
#
# smt_dic = {}
# result = ''
#
# for smt_list in smt_csv:
#     smt_dic[smt_list[1]] = (smt_list[3], smt_list[4])
#     smt_dic[smt_list[2]] = (smt_list[3], smt_list[4])
#
# input_kor = input('입력하세요! : ')
# input_list = input_kor.split(' ')
# for i in range(len(input_list)):
#     for j in input_list:
#         if smt_dic[j][1] == str(i):
#             result = result + smt_dic[j][0] + ' '
#
# print(result)


