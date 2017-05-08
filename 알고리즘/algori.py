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

# p = sorted(list(letter.keys()))
# q = sorted(list(letter.values()))


letter_cnt = int(input())*6
letter_num = input()
temp =[]
start = 0
finish = 6

while finish <= letter_cnt:
    temp.append(letter_num[start:finish])
    start += 6
    finish += 6

paper = []
for key in temp:
    if key in letter.keys():
        paper.append(letter[key])



# for key in letter:
#     if key == p[0]:
#         print(key)

# print(p)
# print(d[0])
# print(temp[0])
print(paper)
print(''.join(paper))
# print(''.join(d)[0])