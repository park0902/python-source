labels = []

with open('D:\\park\\Document_Analysis\\SMSSpamCollection', 'r', encoding='UTF8') as file_handle:
    for line in file_handle:    # 파일을 1줄씩 읽기
        splits = line.split()
        label = splits[0]

        if label == 'spam':     # 맨 앞 단어(label)가 spam 이면 1, 아니면 0
            labels.append(1)
        else:
            labels.append(0)

print(labels)