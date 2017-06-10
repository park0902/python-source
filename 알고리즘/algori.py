def ugly(n):
    result = [1]
    while True:
        last = result[-1]

        if len(result) == n:
            return result

        temp = []

        for r in result:
            print(r)
            print('======')
            for t in r*2,r*3,r*5:
                if t > last:
                    temp.append(t)
                    print(temp)
                    print('------')
        result.append(min(temp))


print(ugly(5))