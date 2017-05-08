import math

def stddv(*args):

    def mean():           # 평균 구하는 함수
        return sum(args)/len(args)

    def variance(m):    # 분산 구하는 함수
        total = 0
        for arg in args:
            total += (arg - m) **2
        return total / (len(args)-1)

    v = variance(mean())      # 분산을 구함
    return math.sqrt(v)     # 분산에 루트를 씌워서 표준편차를 구함

if __name__=="__main__":
    print(stddv(2.3,1.7,1.4,0.7,1.9))