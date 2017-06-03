# list = [1]
#
# for i in range(1,1501):
#     if i%2 == 0 or i%3 == 0 or i%5 == 0:
#         list.append(i)
#
# print(list)




def ugly(n):
    result = [1]
    while True:
        last = result[-1]
        if len(result)==n:
            return result

        tmp = []
        for r in result:
            for t in r*2,r*3,r*5:
                if t > last:
                    tmp.append(t)

        result.append(min(tmp))


print(ugly(1500))



list = [1,2,3,4]

print(min(list))