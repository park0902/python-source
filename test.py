# temp = []
#
# h = int(input())    # 6
# l = int(input())    # 41
# i = 1
#
# while 3*i >= l:
#     for i in range(1,l):
#         temp.append(3*i)
#
# print(temp)

#
# n=100
# a=2;b=7
# print('2', end='')
# while b<=n:
#     print(', %d'%b, end='')
#     a, b = b, a+b



# temp = []
# res = int(input())
# a = int(input())
# b = int(input())
# a;b
# temp.append(a)
# while b <= res:
#     temp.append(b)
#     a, b = b, a+b
#
# print(temp)


# temp_a = []
# temp_b = []
#
# res = int(input())
# a = int(input())
# b = int(input())
# a;b
# temp.append(a)
# while b <= res:
#     temp.append(b)
#     a, b = b, a+b
#
# print(temp)


#



def word_cnt():
    word_script = ''     # 입력받은 단어를 담을 변수
    while word_script != 'end':        # end를 입력하지 않으면 계속 진행
        word_list=[]                   # 입력된 단어 중 중복을 제거하고 남은 단어를 담을 변수
        word_script = str(input('단어를 입력하세요!'))
        if word_script != 'end':        # end가 입력되면 진행하지말고 종료
            if len(word_script) < 200:  # 단어 길이가 200 이하로 입력이 되면 실행
                word = word_script.split()      # 입력받은 word_script를 split함수로 구분
                word_list = list(set(word))     # list(set())함수로 담겨진 단어들 중 중복 제거
                word_list.sort()               # sort()함수로 정렬
                for i in word_list:
                    word_cnt = 0
                    word_cnt += word.count(i) # 중복 제거한 word_list에서 word에 몇개가 들어가있는지 count
                    print(i,':',word_cnt)

print(word_cnt())