# don=[]
# a = input('단위를 입력하세요 띄어쓰기로 구분 ')
# b = a.split()   #입력받은 단어를 split으로 구분해줍니다.
#
# for i in b:
#     don.append(int(i))                # 비워둔 don에 숫자로 변환후에 append함수로 넣어줍니다.
#     don.sort(reverse=True)          # sort함수를 이용해서 큰 숫자 부터 정렬합니다.
# print(don)              #정렬 확인
#
# money = int(input('금액을 입력하세요 '))
#
# for j in don:                              # 큰 숫자부터 정리된 don리스트를 for문에 넣습니다.
#     tt = divmod(money,j)                # 입력받은 금액을 큰 금액 순으로 divmod를 실행해 tt에 넣어줍니다.
#     money = tt[1]                       # tt에 들어간 리스트에서 나머지 값이 담긴 tt[1]을 다시 money로 넣어줍니다.
#     print(j,tt[0],'개가 필요합니다')  #금액과 나눈 몫tt[0]을 출력후 다시 남아있는 금액과
#                                        # 나눌값이 잇으면 다시 loop문으로 올라갑니다.

# for i in range(1,11):
#     if i == 5:
#         continue
#
#     print(i)


# def break_fun(var):
#     list = ''
#     num = 1
#     while(True):
#         list = list + str(num) + '  '
#         if num == var:
#             break
#         num +=1
#     print(list)
#
# print(break_fun(10))


# import csv
#
# emp = []
# emp_file = open("d:\data\emp2.csv", "r")
# emp_csv = csv.reader(emp_file)
#
# for i in emp_csv:
#     emp.append({'empno':i[0], 'ename':i[1], 'job':i[2], 'mgr':i[3],
#                       'hiredate':i[4], 'sal':i[5], 'comm':i[6], 'deptno':i[7]})
#
# print(emp)



# import csv
#
# emp = []
# emp_file = open("d:\data\emp2.csv", "r")
# emp_csv = csv.reader(emp_file)
#
# for i in emp_csv:
#     emp.append({'empno':i[0], 'ename':i[1], 'job':i[2], 'mgr':i[3],
#                 'hiredate':i[4], 'sal':i[5], 'comm':i[6], 'deptno':i[7]})
#
# for emp_dic in emp:
#     print(emp_dic['ename'], emp_dic['sal'], emp_dic['job'])



# import csv
#
# emp = []
# dept = []
# emp_file = open("d:\data\emp2.csv")
# dept_file = open("d:\data\dept.csv")
# emp_csv = csv.reader(emp_file)
# dept_csv = csv.reader(dept_file)
#
# for i in emp_csv:
#     emp.append({'empno':i[0], 'ename':i[1], 'job':i[2], 'mgr':i[3],
#                 'hiredate':i[4], 'sal':i[5], 'comm':i[6], 'deptno':i[7]})
#
# for j in dept_csv:
#     dept.append({'deptno':j[1], 'dname':j[2], 'loc':j[3]})


# import csv
#
# emp = []
# dept = []
# emp_file = open("d:\data\emp2.csv")
# dept_file = open("d:\data\dept.csv")
# emp_csv = csv.reader(emp_file)
# dept_csv = csv.reader(dept_file)
#
# for i in emp_csv:
#     emp.append({'empno':i[0], 'ename':i[1], 'job':i[2], 'mgr':i[3],
#                 'hiredate':i[4], 'sal':i[5], 'comm':i[6], 'deptno':i[7]})
#
# for j in dept_csv:
#     dept.append({'deptno':j[1], 'dname':j[2], 'loc':j[3]})
#
#
# for e in emp:
#     for d in dept:
#         if (e['deptno'] == d['deptno']) & (d['loc'] == 'DALLAS'):
#             print(e['ename'], d['loc'])


# import csv
#
# emp = []
# dept = []
# emp_file = open("d:\data\emp2.csv")
# dept_file = open("d:\data\dept.csv")
# emp_csv = csv.reader(emp_file)
# dept_csv = csv.reader(dept_file)
#
# for i in emp_csv:
#     emp.append({'empno':i[0], 'ename':i[1], 'job':i[2], 'mgr':i[3],
#                 'hiredate':i[4], 'sal':i[5], 'comm':i[6], 'deptno':i[7]})
#
# for j in dept_csv:
#     dept.append({'deptno':j[1], 'dname':j[2], 'loc':j[3]})
#
#
# for e in emp:
#     for d in dept:
#         if (e['deptno'] == d['deptno']) & (d['loc'] == 'DALLAS'):
#             print(e['ename'], d['loc'])
#
# def join(table1, col1, table2, col2, conn_col):
#     for i in table1:
#         for j in table2:
#             if i[conn_col] == j[conn_col]:
#                 print(i[col1], j[col2])
#
# print(join(emp, 'ename', dept, 'loc', 'deptno'))


# import pandas as pd
#
# emp = pd.read_csv("d:\data\emp.csv")
# dept = pd.read_csv("d:\data\dept.csv")
#
# res = pd.merge(emp, dept, on='deptno')
# result = res[['ename','loc']][res['loc'] == 'DALLAS']
#
# print(res)


# import pandas as pd
#
# emp = pd.read_csv("d:\data\emp.csv")
# dept = pd.read_csv("d:\data\dept.csv")
#
# res = pd.merge(emp, dept, on='deptno', how='right')
# result = res[['ename','loc']]
#
# print(result)


# import pandas as pd
#
# def find_loc(ename):
#     emp = pd.read_csv("d:\data\emp.csv")
#     dept = pd.read_csv("d:\data\dept.csv")
#
#     res = pd.merge(emp, dept, on='deptno')
#     result = res[['loc']][res['ename'] == ename]
#
#     return  result
#
# print(find_loc('SMITH'))


# def minbon_fun(num):
#
#     delta = 0.0001
#     result = ((2*(num+delta)*(num+delta)+1)-(2*num*num+1))/delta
#     return result
#
# print(minbon_fun(-2))
