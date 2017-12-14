import numpy as np

vocabulary = {}     # 딕셔너리 선언
with open('D:\\park\\Document_Analysis\\SMSSpamCollection', encoding='UTF8') as file_handle:     # 파일 열기
    for line in file_handle:        # 파일을 1줄씩 읽기
        splits = line.split()       # 한 줄을 공백으로 쪼개서 리스트 만들기
        label = splits[0]           # 맨 앞의 단어는 레이블이니까 따로 변수에 담는다
        text = splits[1:]           # 맨 앞의 단어를 제외한 나머지는 텍스트 변수에 담는다

        # 전체 내용을 단어 단위로 살피고 사전에 해당 단어가 없으면 추가 후 고유번호를 붙인다
        # 그리고 그 매핑을 vocabulary에 저장한다( { '단어' -> '고유ID' } )

        for word in text:
            lower = word.lower()    # 소문자
            if not word in vocabulary:
                vocabulary[word] = len(vocabulary)

# print(vocabulary)
# 각 문서의 피처 벡터를 뽑아서 features 리스트에 넣습니다
features = []
with open('D:\\park\\Document_Analysis\\SMSSpamCollection', encoding='UTF8') as file_handle:
    for line in file_handle:
        splits = line.split()       # 한 줄을 공백으로 쪼개서 리스트 만들기
        text = splits[1:]
        feature = np.zeros(len(vocabulary))
        for word in text:
            lower = word.lower()
            # vocabulary에 따라 각 피처가 몇 번 나왔는지 개수 세기
            feature[vocabulary[lower]] += 1

        # 단어 빈도 피처이므로 문서에서 나온 총 단어 수로 전체 벡터를 나누어 피처를 만든다
        feature = feature / sum(feature)
        features.append(feature)


