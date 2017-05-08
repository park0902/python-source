letter = {'A':'000000', 'B':'001111', 'C':'010011',
          'D':'011100', 'E':'100110', 'F':'101001',
          'G':'110101', 'H':'111010'}

for values in letter:
    print(values, letter[values])




# word = input().split(' ')
# d = {}
#
# while word != 'END':
#     word = input().split(' ')
#     for i in word:
#         if i not in d:
#             d[i] = 1
#
#         elif i in d:
#             d[i] += 1
#
#         elif i == 'END':
#             break
# print(d)



# word = 'brontosaurus'
# d = dict()
#
# for i in word:
#     if i not in d:
#         d[i] = 1
#     else:
#         d[i] = d[i] + 1
#
# print(d)


# word = 'brontosaurus'
# d = dict()
#
# for i in word:
#     d[i] = d.get(i,0) + 1
#
# print(d)