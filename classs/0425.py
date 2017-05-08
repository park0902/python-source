# import math
#
# def stddv(*args):
#
#     def mean():         # 평균 구하는 함수
#         return sum(args)/len(args)
#
#     def variance(m):    # 분산 구하는 함수
#         total = 0
#         for arg in args:
#             total += (arg - m) **2
#         return total / (len(args)-1)
#
#     v = variance(mean())    # 분산을 구함
#     return math.sqrt(v)     # 분산에 루트를 씌워서 표준편차를 구함
#
# print(stddv(2.3,1.7,1.4,0.7,1.9))


# def coinGreedy(money, cash_type):
#     # money = 362 cash_type = [100, 50, 1]
#     #                  idx =    0   1   2
#     def coinGreedyRecursive(money, cash_type, res, idx):
#
#         if idx >= len(cash_type):   # 화폐 다 사용 시 종료
#             # len(cash_type) = 3
#             # if 3 >= 3
#             return res
#
#         dvmd = divmod(money,cash_type[idx])
#         # divmod(362,100) => (3, 62)
#         # divmod(62,50)=> (1, 12)
#         # divmod(12,1) => (12, 0)
#         res[cash_type[idx]] = dvmd[0]   # 해당 화폐 사용 매수
#         # res[100] = 3
#         # res[50] = 1
#         # res[1] = 12
#         return  coinGreedyRecursive(dvmd[1],cash_type,res,idx+1)
#         # dvmd[1] -> 지불하고 남은 금액
#
#         # coinGreedyRecursive(362,[100,50,1],res,0) => 1
#         # coinGreedyRecursive(62,[100,50,1],res,1) => 2
#         # coinGreedyRecursive(12,[100,50,1],res,2) => 3
#         # coinGreedyRecursive(0,[100,50,1],res,3) => 4
#     cash_type.sort(reverse=True)  # 화폐 내림차순 정렬
#     return coinGreedyRecursive(money,cash_type,{},0)
#
# money = int(input('액수입력 : '))
# cash_type = [int(x) for x in input('화폐단위를 입력하세요 : ').split(' ')]
# res = coinGreedy(money,cash_type)
# for key in res:
#     print('{0}원 : {1}개'.format(key,res[key]))