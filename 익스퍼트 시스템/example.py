# import math as m
#
# a = -(0.8*m.log(0.8,2)) -+(0.2*m.log(0.2,2))
# print(a)

# import math as m
#
# def entropy(x, y):
#     result = -(x*m.log(x,2)) -+ (y*m.log(y,2))
#     return result
#
# print(entropy(0.8, 0.2))



# import math as m
#
# def entropy(p2):
#     result = sum(-p * m.log(p,2) for p in p2 if p)
#     return result
#
# print(entropy([4/5, 1/5]))
# print(entropy([0]))


# import math as m
# import collections
#
# card_yn = ['Y', 'Y', 'N', 'Y', 'Y']
#
# def entropy(p2):
#     return sum(-p * m.log(p,2) for p in p2 if p)
#
# def class_probailities(labels):
#     total = len(labels)
#     return [i/total for i in collections.Counter(labels).values()]
#
# card = class_probailities(card_yn)
#
# print(entropy(card))


'''


import math as m
import collections

inputs = [({'cust_name':'SCOTT', 'card_yn':'Y', 'review_yn':'Y', 'before_buy_yn':'Y'}, True),
          ({'cust_name':'SMITH', 'card_yn':'Y', 'review_yn':'Y', 'before_buy_yn':'Y'}, True),
          ({'cust_name':'ALLEN', 'card_yn':'N', 'review_yn':'N', 'before_buy_yn':'Y'}, False),
          ({'cust_name':'JONES', 'card_yn':'Y', 'review_yn':'N', 'before_buy_yn':'N'}, True),
          ({'cust_name':'WARD',  'card_yn':'Y', 'review_yn':'Y', 'before_buy_yn':'Y'}, True)]

def class_probailities(labels):
    total = len(labels)
    return [i/total for i in collections.Counter(labels).values()]

def entropy(p2):
    return sum(-p * m.log(p,2) for p in p2 if p)

def columndata(inputs, key):
    groups = []
    for input in inputs:
        groups.append(input[0][key])
    return groups

for i in ['card_yn', 'review_yn', 'before_buy_yn']:
    yn = columndata(inputs, i)
    p = class_probailities(yn)
    print(i, entropy(p))


'''





'''

from collections import defaultdict

group1 = {}
group1['one'] = 'a'

def noname():
    return 'a'

group2 = defaultdict(noname)
group2['one']

group3 = defaultdict(lambda:'a')

group4 = defaultdict(list)  # 비어있는 list 를 default 값으로 하겠다
group4['Y']

print(group1['one'])
print(group2['one'])
print(group3['one'])
print(group4['Y'])

'''



'''
- 구매여부 데이터에 대한 분할후 엔트로피 출력하는 코드
'''

import math     # 엔트로피 계산에 로그함수를 쓰기 위해 math 모듈을 불러온다.
import collections
from collections import defaultdict


def class_probabilities(labels):       # 엔트로피 계산에 사용하는 컬럼의 확률을 계산하는 함수
    total_count = len(labels)
    return [count / total_count for count in collections.Counter(labels).values()]


def entropy(class_probabilities):
    return sum(-p * math.log(p, 2) for p in class_probabilities )


inputs = [( {'cust_name':'SCOTT', 'card_yn':'Y', 'review_yn':'Y', 'before_buy_yn':'Y'}, True),
          ( {'cust_name':'SMITH', 'card_yn':'Y', 'review_yn':'Y', 'before_buy_yn':'Y'}, True),
          ( {'cust_name':'ALLEN', 'card_yn':'N', 'review_yn':'N', 'before_buy_yn':'Y'}, False),
          ( {'cust_name':'JONES', 'card_yn':'Y', 'review_yn':'N', 'before_buy_yn':'N'}, True),
          ( {'cust_name':'WARD',  'card_yn':'Y', 'review_yn':'Y', 'before_buy_yn':'Y'}, True) ]


def partition_by(inputs, attribute):
    groups = defaultdict(list)
    for input in inputs:
        key = input[0][attribute]
        groups[key].append(input)
    return groups

for key in ['card_yn','review_yn','before_buy_yn']:
     print(partition_by(inputs,key))

def data_entropy(labeled_data):
    labels = [label for _, label in labeled_data]
    # print(labels)
    probabilities = class_probabilities(labels)
    #print (entropy(probabilities))
    return entropy(probabilities)


def partition_entropy(subsets):         # 파티션된 노드들의 엔트로피
    total_count = sum(len(subset) for subset in subsets)        # subset은 라벨이 있는 데이터의 리스트의 리스트이다. 이것에 대한 엔트로피를 계산한다.
    # for subset in subsets:
    #     print(len(subset))
    #     print(data_entropy(subset))
    return sum(data_entropy(subset) * len(subset) / total_count for subset in subsets)


for key in ['card_yn','review_yn','before_buy_yn']:
    #print( partition_by(inputs,key).values())
    print(key, partition_entropy(partition_by(inputs, key).values()))






