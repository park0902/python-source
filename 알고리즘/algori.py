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

p = sorted(list(letter.keys()))
q = sorted(list(letter.values()))


letter_cnt = int(input())*6
letter_num = input()
temp =[]
# temp = ['001111', '000000', '011100']
start = 0
finish = 6
cnt = 0
result = []

while finish <= letter_cnt:
    temp.append(letter_num[start:finish])
    start += 6
    finish += 6

for key in temp:
    if key in letter.keys():
        result.append(letter[key])

    if key not in letter.keys():
        for i in range(6):
            for j in range(6):
                if key[j] == p[i][j]:
                    cnt += 1






# for key in letter:
#     if key == p[0]:
#         print(key)

print(result)
# print(p)
# print(d[0])
# print(temp[0])
# print(''.join(result))
# print(''.join(paper))
# print(''.join(d)[0])