# a = input('숫자를 입력하시오! : ')
#
# if int(a)%2 == 0:
#     print('짝수')
#
# else:
#     print('홀수')

# def mod(val):
#     if val%2 == 0:
#         print('짝수')
#     else:
#         print('홀수')
#
# print(mod(10))


# a = input('이름을 입력하시오! ')
#
# import pandas as pd
# emp = pd.read_csv("d:\data\emp.csv")
# salary = emp[['sal']][emp['ename'] == a].values[0]
#
# if salary >= 3000:
#     print('고소득자')
#
# elif salary >= 2000:
#     print('적당')
#
# else:
#     print('저소득자')


# num1 = int(input('첫번째 숫자 입력! '))
# num2 = int(input('두번째 숫자 입력! '))
#
# sub1 = (num1 + num2)/2
# sub2 = (num2 - num1 + 1)
# result = sub1 * sub2
#
# print(num1,'부터',num2,'까지의 숫자 합은 ',result,'입니다')


# num1 = int(input('첫번째 숫자 입력! '))
# num2 = int(input('두번째 숫자 입력! '))
#
# if num1 < num2:
#     sub1 = (num1 + num2) / 2
#     sub2 = (num2 - num1 + 1)
#     result = sub1 * sub2
#     print(num1,'부터',num2,'까지의 숫자 합은 ',result,'입니다')
#
# else:
#     print('첫번째 입력한 숫자가 두번째 입력한 숫자보다 큽니다')


# for i in range(2,10):
#     for j in range(1,10):
#         print(i,' X ',j,' = ',i*j)
#     print('\n')

# for i in range(1,11):
#     print('★'*i)

# def lpad(val,num,st):
#     return st * (num-len(val)) + val
#
# for i in range(1,5):
#     print(lpad('★',8-i,'   ')*i)
# for j in range(3,0,-1):
#     print(lpad('★',8-j,'   ')*j)

# for i in range(11):
#     if i < 6:
#         print(' '*(20-i),'★'*i)
#
#     elif i >= 6:
#         print(' '*(20-i),'★'*(i-5),' '*(12-i),'★'*(i-5))


# for i in range(1,10):
#     p = ''
#     for j in range(2,10):
#         p += (str(j)+' X '+str(i)+' = '+str(j*i)).ljust(16)
#
#     print(p)



# s = 'some string12'
# number = sum(i.isdigit() for i in s)
# word = sum(i.isalpha() for i in s)
# space = sum(i.isspace() for i in s)
#
# print(number)
# print(word)
# print(space)


# text_file = open("d:\data\winter.txt","r")
# lines = text_file.readlines()
# total1 = 0
# total2 = 0
#
# for s in lines:
#
#     number = sum(i.isdigit() for i in s)
#     word = sum(i.isalpha() for i in s)
#     space = sum(i.isspace() for i in s)
#     cnt = len(s)
#     total1 += cnt
#     total2 += (number+word+space)
#
# print(total1-total2)


# print(num1,' 의',num2,'승은 ',res,' 입니다')

# ga = int(input('가로 숫자 입력! '))
# se = int(input('세로 숫자 입력! '))
#
# for i in range(se):
#     res = ''.join('★' for i in range(ga))
#     print(res)


# s = int(input('팩토리얼 숫자를 입력하시오! '))
# result = 1
# count = 0
#
# while count < s:
#     result = result * s
#     s = s - 1
#
# print(result)

# base = int(input('밑수 입력! '))
# exp = int(input('진수 입력! '))
# cnt = 0
#
# while exp >= base:
#
#     exp = exp/base
#     cnt = cnt + 1
#
# print(cnt)

