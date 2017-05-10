import os
import psutil
import time

###################################################################################################
## 1. 문제        : 오류고정 (고급)
## 2. 소요 시간   : 0.0 초 (소수점 6자리 반올림)
## 3. 사용 메모리 : 163840 byte
## 4. 만든 사람   : 길용현
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

p = list(letter.keys())



letter_cnt = int(input())*6
letter_num = input()

# 시작 시간 체크
stime = time.time()



temp =[]
# temp = ['001111', '000000', '011100']
start = 0
finish = 6
cnt = 0
# result = ''
result = []
while finish <= letter_cnt:
    temp.append(letter_num[start:finish])
    start += 6
    finish += 6

# print(p)
count = 0
for key in temp:
           for i in range(8):
            cnt = 0
            # count = 0

            for j in range(6):
                # print(key)

                if key[j] == p[i][j]:
                    cnt += 1
                    # print(count)
                    # print(key[j],'-',p[i][j])
                    # print(cnt)
                    if cnt == 5:
                        count += 1
                        result.append(letter[p[i]])
                        # print(cnt)

                    # elif cnt < 5:




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