import os
import psutil
import time

###################################################################################################
## 1. 문제        : 비밀편지
## 2. 소요 시간    : 0.0 초 (소수점 6자리 반올림)
## 3. 사용 메모리  : 28672 byte
## 4. 만든 사람    : 박상범
###################################################################################################

# 시작 메모리 체크
proc1 = psutil.Process(os.getpid())
mem1 = proc1.memory_info()
before_start = mem1[0]

# 001111000000011100
letter = {
          '000000':'A',
          '001111':'B',
          '010011':'C',
          '011100':'D',
          '100110':'E',
          '101001':'F',
          '110101':'G',
          '111010':'H'
         }

# 비밀편지 딕셔너리 키 값 리스트화
letter_key = list(letter.keys())


letter_cnt = int(input())*6     # 문자의 개수
letter_num = input()            # 비밀편지 숫자

# 시작 시간 체크
stime = time.time()


temp =[]        # 6개씩 자른 비밀편지 숫자를 담을 리스트 변수
start = 0       # 비밀편지 숫자를 6개씩 자르기 위한 변수1
finish = 6      # 비밀편지 숫자를 6개씩 자르기 위한 변수2
cnt = 0         # 위치 출력을 위한 변수
result = []     # 6개씩 자른 숫자를 문자로 담을 리스트 변수
# result = ''

while finish <= letter_cnt:
    temp.append(letter_num[start:finish])
    start += 6
    finish += 6

count = 0

for key in temp:
    for i in range(8):
        cnt = 0
            # count = 0

        for j in range(6):
                # print(key)

            if key[j] == letter_key[i][j]:
                cnt += 1
                    # print(count)
                    # print(key[j],'-',p[i][j])
                    # print(cnt)
                if cnt == 5:
                    count += 1
                    result.append(letter[letter_key[i]])
                        # print(cnt)




# for key in temp:
#     if key in letter.keys():
#         result += letter[key]
#         # print(result,'=')
#     elif key not in letter.keys():
#         for i in range(len(key)):
#             cnt = 0
#             ct = 0
#             for j in range(len(key)):
#                 # print(key)
#                 if key[j] == p[i][j]:
#                     cnt += 1
#                     ct += 1
#                     # print(key[j],'-',p[i][j])
#                 if cnt == 5:
#                     result += letter[p[i]]
#                 #
#                 elif cnt <= 4:
#                     result = str(ct)



# for key in letter:
#     if key == p[0]:
#         print(key)

print(''.join(result))
print(count)


# 종료 시간 체크
etime = time.time()
print('consumption time : ', round(etime-stime, 6))

# 실행 후 맨 밑에서 코드 구동 후 메모리 체크
proc = psutil.Process(os.getpid())
mem = proc.memory_info()
after_start = mem[0]
print('memory use : ', after_start-before_start)


# print(p)
# print(d[0])
# print(temp[0])
# print(''.join(result))
# print(''.join(paper))
# print(''.join(d)[0])