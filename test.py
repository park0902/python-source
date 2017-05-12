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


def fi_b(num):
    res = [1,2]

    for i in range(2, num-2):
        res.append(res[i-1]+res[i-2])

    return res

def fi_a(num):
    res = [1,1]

    for i in range(2, num-2):
        res.append(res[i-1]+res[i-2])

    return res
#
last = int(input())
value = int(input())

a = fi_a(last)[-1]
b = fi_b(last)[-1]
c = fi_a(last)
d = fi_b(last)

# print(c)
# print(d)

p = []
q = []
s = ''
t = ''

for i in range(1,value):
    if a*i <= value:
        p.append(a*i)

for i in range(1,value):
    if b*i <= value:
        q.append(b*i)

# print(p)
# print(q)
# print('='*10)
for i in p:
    for j in range(len(p)):
        for k in range(len(q)):
            # print(p[j]+q[k], j+1, k+1)
            if p[j]+q[k] == value and j < k:
                s = j+1
                t = k+1

print(s)
print(t)