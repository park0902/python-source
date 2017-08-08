# Music
from librosa import load, stft, feature, get_duration, display as dp
from sklearn.metrics.pairwise import cosine_similarity
from pygame import mixer, init, display, time, quit
from ctypes import windll
from random import choice
import matplotlib.pyplot as plt
import numpy as np
import os

################## 플레이 정보 ##################
play_duration = 10                # 노래 재생 시간
file_loc = 'D:\data\music' # 노래 폴더 위치
show_chroma = False               # 크로마그램 출력
###############################################


################## 노래 클래스 ##################
class Song(object):
    def __init__(self,show=show_chroma):
        self.music_dict = {}  # 노래 이름을 key로, 파일경로를 value로 하는 딕셔너리
        self.music = ''       # 재생할 노래 이름
        self.time = 0         # 노래 총 길이
        self.show = show      # 크로마그램 화면에 출력할지 말지 여부
        self.best_repeated_part = []      # 최종 하이라이트 시간 정보를 담는 변수
        self.chroma_len = 0

    ##### 노래 로딩 메소드 #####

    # (노래 로딩 Main) 노래 로드 메인 메소드(여기서 리턴한 노래의 코사인 유사도 정보를 Analysis() 메소드에서 분석)
    def _LoadSong(self):
        self.music_dict = self._SearchSong(file_loc)  # 노래, 파일경로 딕셔너리 생성
        self.music = self._Input()                    # 재생할 노래 이름 입력

        if self.music.upper() == 'RANDOM':           # 랜덤 입력시 노래 리스트 중 하나 랜덤으로 뽑기
            self.music = choice(list(self.music_dict.keys()))

        y, sr = load(self.music_dict[self.music], sr=882) # librosa 모듈의 노래 불러오기(샘플링은 0.5초 당 하나씩)
        s = np.abs(stft(y)**2)                            # 노래의 파워 데이터(주파수의 진폭)
        self.time = get_duration(y=y, sr=sr)              # 노래의 총 길이(샘플링 데이터 갯수)
        chroma = feature.chroma_stft(S=s, sr=sr)          # 크로마그램으로 변환
        self._Chromagram(chroma, show=self.show, title='music chromagram_line41')  # 크로마그램 그래프를 출력
        chromaT = np.transpose(chroma,axes=(1,0))         # time-time 코사인 유사도 구하기 위해 전치행렬로 변환
        self._Chromagram(chromaT, show=self.show, title='music transpose_line43')  # 전치화된 크로마그램 그래프를 출력
        print('\nLoading Finished!')
        return cosine_similarity(chromaT)                 # 노래 각 부분의 코사인 유사도를 리턴

    # (노래 로딩 SUB) 폴더에서 mp3 파일 리스트를 뽑아서 딕셔너리에 담는 메소드
    def _SearchSong(self, dirname):
        filenames = os.listdir(dirname)                     # 지정된 폴더 내 파일이름들 불러오기
        music_dict = {}
        for filename in filenames:
            full_filename = os.path.join(dirname, filename) # full_filename = 경로+파일이름
            ext = os.path.splitext(full_filename)[-1]       # ext에 확장자 넣기
            file = os.path.splitext(filename)[0]            # file에 확장자를 제외한 파일이름만 넣기
            if ext == '.mp3':                               # 확장자가 mp3 인 파일만 music_dict 딕셔너리에 넣기
                music_dict[file] = full_filename            # 파일이름(key), 경로+파일이름(value)
        return music_dict                                   # music_dict 딕셔너리 리턴

    # (노래 로딩 SUB) 처음에 노래 리스트를 화면에 출력하고 재생할 노래 입력받는 메소드
    def _Input(self):
        print('---------------------------------------------------'
              'Music List---------------------------------------------------')
        music_dict_list = list(self.music_dict.keys())
        for idx in range(len(music_dict_list)//5 + 1):
            try:
                print(' '.join([i.ljust(25) for i in music_dict_list[5 * idx : 5 * idx +5]]))
            except IndexError:
                print(' '.join([i.ljust(25) for i in music_dict_list[5 * idx: -1]]))

        return input('\n원하는 노래 제목을 입력하세요.(랜덤 원할 경우 random 입력) ')

    ##### 노래 분석 메소드 #####

    # (노래 분석 MAIN) 하이라이트를 뽑아서 리턴하는 메소드(여기서 리턴한 하이라이트 정보를 Play클래스의 PlaySong()메소드에서 재생)
    def Analysis(self):
        chroma = self._Denoising()                 # 코사인유사도 정보를 Denoising()메소드를 이용해 노이즈 제거
        self._Chromagram(chroma, show=self.show, title='final_line77')   # self.show = True 일 경우 노래 시작 전 크로마그램 그래프를 출력
        return self.best_repeated_part[0] - 1.5               # 하이라이트 시간(self.result[0])-1.5초를 리턴(하이라이트 직전부터 재생)

    # (노래 분석 SUB) 코사인 유사도의 노이즈를 제거하는 메소드(Filtering()메소드와 Tensor()메소드를 이용해 노이즈 제거)
    def _Denoising(self):
        chroma = self._LoadSong()                  # 코사인유사도를 chroma에 넣기
        self._Chromagram(chroma, show=self.show, title='after cosine similarity_line83') # 코사인 유사도 적용 후 크로마그램
        converttime = (self.time / len(chroma))   # 샘플링된 노래 시간 정보를 실제 노래 시간 정보로 변환해주기 위한 변수
        filtered_chroma = self._Filtering(chroma)  # Filtering()메소드로 chroma 의 노이즈 제거
        filterrate = 0.25                         # Filtering()메소드의 필터링 비율(하위25% 값을 제거한다는 의미)
        while filtered_chroma.all() == 0 :        # 필터링 비율이 높아 전체 값이 제거되는 경우 필터링 비율을 낮춰줌
            filtered_chroma = self._Filtering(chroma, filterrate= filterrate-0.05)
        self._FindBestParts(filtered_chroma, converttime) # 필터링된 chroma 를 FindBestParts()메소드에 넣어 하이라이트 찾기
        return filtered_chroma                    # 필터링된 chroma를 리턴

    # (노래 분석 SUB) 가장 많이 반복된 부분을 찾아서 하이라이트로 지정하는 메소드
    def _FindBestParts(self,chroma,converttime):   #
        for rn in range(len(chroma)):          # chroma에서 행과 열이 같은 부분(중복 부분)제거하여 노이즈 추가 제거
            for cn in range(rn):
                chroma[rn][cn] = 0
        chroma[chroma <= 0] = 0                # 노이즈 추가 제거 2
        repeatedcnt = self._LineFilter(chroma)  # LineFilter()메소드 사용해 시간대별 반복횟수 카운트한 리스트 생성
        best_repeated = round(converttime * max(repeatedcnt, key = lambda item: item[1])[0],1) # 최다 반복된 부분 선택
        print('\nMusic Name : {}'.format(self.music.upper()))
        print('Highlight : {}m {}s'.format(int(best_repeated//60), int(best_repeated%60)))
        self.best_repeated_part.append(best_repeated)  # best_repeated_part 에 하이라이트 부분 넣기

    ##### 노래 필터링 메소드 #####

    # (필터링1 MAIN) 노이즈 제거 시 쓰는 필터링 메소드
    def _Filtering(self, chroma, cnt=3, filterrate = 0.25):
        recursive_cnt = cnt                                  # 재귀횟수
        if recursive_cnt == 3 :
            self.chroma_len = len(chroma)
        chroma = np.pad(chroma, pad_width=4, mode='constant', constant_values=0)
        chroma = Common.im2col_sliding_strided(chroma, (9,9))
        chroma = np.dot(chroma, self._Tensor()) / 9
        chroma = chroma.reshape(self.chroma_len,-1)
        chroma[chroma <= filterrate * np.max(chroma)] = 0    # 노이즈 제거
        self._Chromagram(chroma, show=self.show, title='filtering_line116')  # 필터링한 크로마그램 그래프를 출력

        if recursive_cnt == 0:                               # 마지막 재귀 시 chroma 를 정규화시켜서 데이터 정제하고 리턴
            return Common.Normalization(chroma)

        print('Count down', recursive_cnt)
        return self._Filtering(chroma, cnt=recursive_cnt - 1) # 재귀를 3번 돌리며 필터링

    # (필터링1 SUB) 필터링 시 사용하는 Tensor 를 만드는 메소드
    def _Tensor(self):
        tensor = np.eye(9,9).reshape(-1,1)
        tensor[tensor<=0] = -1/9
        return tensor

    # (필터링2) 코사인 유사도가 높은 대각선 방향의 패턴을 카운팅하는 메소드
    def _LineFilter(self,chroma, mincorrectcnt=3, line=25):
        mincnt = mincorrectcnt  # 최소 일치 횟수(대각선 패턴이 일정 횟수 이상 반복되는 경우에 리턴시키도록 제한하여 필터링 질 향상)
        shorterline = line      # 패턴 반복 확인 시 사용할 기준 대각선의 길이
        repeatedcnt = []        # 시간대별 반복횟수 저장할 리스트

        for cn in range(len(chroma)-line):
            correctcnt = 0      # 패턴 일치 횟수
            for rn in range(len(chroma)-line):
                cnt = 0
                while chroma[rn+cnt][cn+cnt] != 0 and cnt < line:
                    cnt += 1
                if cnt == line: # 패턴 일치할 경우 correctcnt 하나씩 증가
                    correctcnt += 1
            repeatedcnt.append([cn,correctcnt]) # 시간대와 시간대별 일치횟수를 리스트에 저장

        #--- 이하 리턴절은 3가지의 시나리오로 나뉘어짐 ---#

        # 3.기준 대각선의 길이가 5가 될때가지 최소 일치 횟수를 충족시키지 못한경우 재귀하여 최소 일치 횟수를 줄여서 패턴 확인
        if line <= 5 :
            return self._LineFilter(chroma, mincorrectcnt=mincnt-1, line=25)

        # 1.최소 일치 횟수 이상 패턴이 반복될 경우 정상적으로 리턴
        if max(repeatedcnt, key=lambda k:k[1])[1] >= mincnt:
            return repeatedcnt
        # 2.최소 일치 횟수 이상 패턴이 반복되지 않는 경우 기준 대각선의 길이를 5만큼 줄여서 재귀하여 패턴 확인
        return self._LineFilter(chroma, mincorrectcnt=mincnt, line=shorterline-5)

    ##### 크로마그램 출력 메소드 #####

    def _Chromagram(self, chroma, show=False, title=None):
        if show == True:
            plt.figure(figsize=(10, 10))
            dp.specshow(chroma, y_axis='time', x_axis='time')
            plt.colorbar()
            # plt.title('{}'.format(self.music.upper()))
            plt.title(title)
            plt.tight_layout()
            plt.show()

################## 재생 클래스 ##################
class Play(object):
    @staticmethod
    def PlaySong():
        song = Song()
        highlight = song.Analysis()

        init()
        mixer.init()
        display.set_mode((100,100))
        SetWindowPos = windll.user32.SetWindowPos
        SetWindowPos(display.get_wm_info()['window'], -1, 0, 0, 0, 0, 0x0003)
        mixer.music.load(song.music_dict[song.music])
        print('Music Start!')
        mixer.music.play(start=highlight)
        time.wait(play_duration * 1000)
        quit()

################## 기타 함수 클래스 ##################
class Common:
    # 행렬곱하기 좋게 행렬을 변환시켜주는 메소드
    @staticmethod
    def im2col_sliding_strided(A, padsize, stepsize=1):
        m, n = A.shape
        s0, s1 = A.strides
        BSZ = [m + 1 - padsize[0], n + 1 - padsize[1]]
        nrows = m - BSZ[0] + 1
        ncols = n - BSZ[1] + 1
        shp = BSZ[0], BSZ[1], nrows, ncols
        strd = s0, s1, s0, s1

        out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
        return out_view.reshape(BSZ[0] * BSZ[1], -1)[:, ::stepsize]

    # 정규화 메소드
    @staticmethod
    def Normalization(chroma):
        for idx in range(len(chroma)):
            chroma[idx][idx] = 0
        return (chroma-np.mean(chroma))/np.max(chroma)

################## 메인 실행절 ##################
if __name__ == '__main__':
    Play.PlaySong()











# 합성곱 연산
import numpy as np

# 6x6 행렬 만들기
a = np.array([i for i in range(36)]).reshape(6,6)


# 3x3 필터 만들기
Filter = np.eye(3,3)

# 행렬 확인
print('---a\n',a)
print('---filter\n',Filter)

############################################################################

### 합성곱 연산 방법 1 ###

# 단일 곱셈-누산 vs 행렬곱 연산
d = np.array([[1,2,3],[4,5,6],[7,8,9]])
print('---d\n',d)
print('---단일 곱셈-누산 결과\n', np.sum(Filter * d)) # (1 * 1) + (-1 * 2) + (-1 * 3) + (1 * 4)
print('---행렬곱 연산 결과\n', np.dot(Filter, d))

# 넘파이 array indexing
print('---a[:,:]\n',a[:,:])            # a 전체 출력
print('---a[:,1:2]\n',a[:,0:3])      # a의 전체행 / 첫번째열~세번째열 출력
print('---a[0:3,4:5]\n',a[3:5,4:5])    # a의 네번재행~다섯번째행 / 다섯번째열 출력

# 스트라이드
for rn in range(len(a[0])-1):
    for cn in range(len(a[1])-1):
        print('---',[rn,cn],'\n',a[rn:rn+2, cn:cn+2])

# 합성곱 연산
result = []

for rn in range(len(a[0])-2):
    for cn in range(len(a[1])-2):
        result.append(np.sum(a[rn:rn+3, cn:cn+3] * Filter))

print('---result\n',result)
print('---len(result)\n', len(result))
len_a = int(np.sqrt(len(result)))
result = np.array(result).reshape(len_a,len_a)
print('---result.reshape\n', result)

# 패딩
a_pad = np.pad(a, pad_width=1, mode='constant', constant_values=0)
print('---a_pad\n',a_pad)

a_pad2 = np.pad(a, pad_width=2, mode='constant', constant_values=-1) # constant_values로 숫자 변경 가능
print('---a_pad2\n',a_pad2)

a_pad3 = np.pad(a, pad_width=((1,2),(3,4)), mode='constant', constant_values=0) # pad_width=( (위, 아래), (왼쪽, 오른쪽 패드 수) )
print('---a_pad3\n',a_pad3)

# 패딩 적용한 합성곱 연산
result2 = []

for rn in range(len(a_pad[0])-2):
    for cn in range(len(a_pad[1])-2):
        result2.append(np.sum(a_pad[rn:rn+3, cn:cn+3] * Filter))

print('---result2\n',result2)
print('---len(result2)\n', len(result2))
len_a2 = int(np.sqrt(len(result2)))
result2 = np.array(result2).reshape(len_a2,len_a2)
print('---result2.reshape\n', result2)


# 문제(1). 0부터 143까지 원소로 이뤄진 12x12 행렬을 만들고, 4x4 필터(단위 행렬)를 이용해 합성곱을 해보세요.
#         (단, 스트라이드는 1, 출력 행렬은 12x12가 되도록 패딩을 적용하세요)


############################################################################

### 합성곱 연산 2 ###

# 단일 곱셈 누산 -> 행렬곱 연산
print('---Filter\n', Filter)
print('---Filter.flatten()\n', Filter.flatten())  # flatten은 행렬을 벡터로 만들어줌

# 행렬곱하기 좋게 행렬을 변환해주는 함수
def im2col_sliding_strided(A, filtersize, stepsize=1): # A = 변환할 행렬, filtersize = 필터 크기, stepsize = 스트라이드
    m, n = A.shape
    s0, s1 = A.strides
    BSZ = [m + 1 - filtersize[0], n + 1 - filtersize[1]]
    nrows = m - BSZ[0] + 1
    ncols = n - BSZ[1] + 1
    shp = BSZ[0], BSZ[1], nrows, ncols
    strd = s0, s1, s0, s1

    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(BSZ[0] * BSZ[1], -1)[:, ::stepsize]

print('---변환 전 a\n', a)
print('---변환 후 a\n', im2col_sliding_strided(a, [3,3]))

# 행렬곱 연산을 이용한 합성곱

a_pad = np.pad(a, pad_width=1, mode='constant', constant_values=0)
a2 = im2col_sliding_strided(a_pad, [3,3])
Filter2 = Filter.flatten()
result = np.dot(a2, Filter2)
print('---합성곱 결과\n', result)
result = result.reshape(6,6)
print('---최종 결과\n', result)

# 문제(2). 앞에서 배운 두 가지 합성곱 방법을 각각 이용하여 0~1사이의 난수로 이루어진 300x300 행렬을
#          9x9 필터(단위행렬)를 이용해 합성곱을 해보세요. (단, 스트라이드는 1, 출력 행렬 크기는 300x300이 되도록 패딩을 적용하세요)








# 문제(1). 0부터 143까지 원소로 이뤄진 12x12 행렬을 만들고, 4x4 필터(단위 행렬)를 이용해 합성곱을 해보세요.
#         (단, 스트라이드는 1, 출력 행렬은 12x12가 되도록 패딩을 적용하세요)
import numpy as np

a = np.array([i for i in range(144)]).reshape(12,12)

Filter = np.eye(4,4)

a_pad = np.pad(a, pad_width=((2,1),(1,2)), mode='constant', constant_values=0)


# 패딩 적용한 합성곱 연산
result2 = []

for rn in range(len(a_pad)-3):
    for cn in range(len(a_pad)-3):
        result2.append(np.sum(a_pad[rn:rn+4, cn:cn+4] * Filter))


print(np.array(result2).reshape(12,12))





# 0~15까지 4X4 행렬에 제로패딩1 만들기
import numpy as np

x = np.array([i for i in range(16)]).reshape(4,4)

padding = np.pad(x, pad_width=1, mode='constant', constant_values=0)

print(padding)




# 0~15까지 4X4 행렬을 만들고 0~8까지 3X3 필터를 이용해서 합성곱
import numpy as np

x = np.array(range(16)).reshape(4,4)
Filter = np.array(range(9)).reshape(3,3)

result = []

for rn in range(len(x)-2):
    for cn in range(len(x)-2):
        result.append(np.sum(x[rn:rn+3, cn:cn+3] * Filter))

print(np.array(result).reshape(2,2))




# 0~35 6X6 행렬을 만들고 0~15 4X4 필터를 이용해서 합성곱
import numpy as np

x = np.array([range(36)]).reshape(6,6)
Filter = np.array([range(16)]).reshape(4,4)

result = []

for rn in range(len(x)-3):
    for cn in range(len(x)-3):
        result.append(np.sum(x[rn:rn+4, cn:cn+4] * Filter))

print(np.array(result).reshape(3,3))



# 패딩값을 구해주는 함수 생성
def padding(OH, S, H, FH):
    return ((S*(OH-1)) + FH - H)/2

print(padding(9,1,9,4))






# 0~15 4X4 행렬을 만들고 0~8 3X3 필터를 만들고 합성곱(출력 4X4)
import numpy as np

a = np.array(range(16)).reshape(4,4)

Filter = np.array(range(9)).reshape(3,3)

a_pad = np.pad(a, pad_width=1, mode='constant', constant_values=0)


# 패딩 적용한 합성곱 연산
result = []

for rn in range(len(a_pad)-2):
    for cn in range(len(a_pad)-2):
        result.append(np.sum(a_pad[rn:rn+3, cn:cn+3] * Filter))


print(np.array(result).reshape(4,4))




# 0~35 6X6 행렬을 만들고 0~8 3X3 필터를 이용해서 합성곱(출력 6X6)
import numpy as np

a = np.array(range(36)).reshape(6,6)

Filter = np.array(range(9)).reshape(3,3)

a_pad = np.pad(a, pad_width=1, mode='constant', constant_values=0)


# 패딩 적용한 합성곱 연산
result = []

for rn in range(len(a_pad)-2):
    for cn in range(len(a_pad)-2):
        result.append(np.sum(a_pad[rn:rn+3, cn:cn+3] * Filter))


print(np.array(result).reshape(6,6))



# 0~24 5X5 행렬을 만들고 0~3 2X2 필터를 이용해서 합성곱(출력 5X5)
import numpy as np

a = np.array(range(25)).reshape(5,5)

Filter = np.array(range(4)).reshape(2,2)

a_pad = np.pad(a, pad_width=((1,0),(0,1)), mode='constant', constant_values=0)


# 패딩 적용한 합성곱 연산
result = []

for rn in range(len(a_pad)-1):
    for cn in range(len(a_pad)-1):
        result.append(np.sum(a_pad[rn:rn+2, cn:cn+2] * Filter))


print(np.array(result).reshape(5,5))




# 스트라이드와 패딩
import numpy as np

class CNN:
    def __init__(self):
        self.input = None
        self.filter = None
        self.output_size = None
        self.x_pad = None
        self.stride = None

    def calCulateOutputSize(self, x, flt, pad=False, stride=1):
        self.input = x
        self.filter = flt
        self.stride = stride
        ##size 계산

        # padding 적용 안했을 때
        input_size = self.input.shape[0]
        filter_size = self.filter.shape[0]
        if pad is False:
            self.output_size = ((input_size-filter_size)/stride)+1
            return input_size,self.output_size

        # padding 적용했을 때
        else:
            x_pad = np.pad(self.input, pad_width=((1,2),(2,1)), mode = 'constant')  # padding 적용
            self.x_pad = x_pad
            x_pad_size = x_pad.shape[0]  # 9x9로 변환
            print(x_pad.shape)
            self.output_size = ((x_pad_size-filter_size)/stride)+1
            return input_size,self.output_size

    def adJustPad(self):

        #패딩 적용했을경우 추출하기
        result = []
        f = self.filter
        f_size = f.shape[0]
        stride = self.stride
        output_size = int(self.output_size)


        for i in range(0,output_size,stride):
            for j in range(0,output_size,stride):
                temp = self.x_pad[i:(f_size+i),
                       j:(f_size+j)]
                result.append(np.sum(temp*f))
        return self.x_pad, result


# 입력값 생성 (7,7)
x = np.arange(49)
x = x.reshape(7,7)

# 임의의 필터 생성 (3,3)
flt = np.array([[4,2,0],[3,2,6],[2,6,2]])

cnn = CNN()
cnn2 = CNN()


# 패딩 적용하지 않은 상태
just_input_size, just_output_size = cnn2.calCulateOutputSize(x,flt, pad=False)
pad_input_size, pad_output_size = cnn.calCulateOutputSize(x,flt, pad=True, stride=1)

# print('패딩 적용 안했을 때',just_input_size==just_output_size)
# print('패딩 적용 후 ',pad_input_size==pad_output_size)



x_pad, result = cnn.adJustPad()
print(x)
print(x_pad)

print(pad_output_size)
p = int(pad_output_size)
print(np.array(result).reshape(p,-1))



#
import numpy as np

x = np.array([[1,2,0,0],[0,1,-2,0],
               [0,0,1,2],[2,0,0,1]])

x1 = np.array([[1,0,0,0],[0,0,-2,-1],
               [3,0,1,0],[2,0,0,1]])

f = np.array([[-1,0,3],[2,0,-1],[0,2,1]])
f1 = np.array([[0,0,0],[2,0,-1],[0,-2,1]])




# 패딩 적용한 합성곱 연산
result = []

for rn in range(len(x)-2):
    for cn in range(len(x)-2):
        result.append(np.sum(x[rn:rn+3, cn:cn+3] * f) + np.sum(x1[rn:rn+3, cn:cn+3] * f1))


print(np.array(result).reshape(2,2))



import numpy as np

x = np.array([[[1,2,0,0], [0,1,-2,0], [0,0,1,2], [2,0,0,1]],[[1,0,0,0], [0,0,-2,-1], [3,0,1,0], [2,0,0,1]]])
f = np.array([[[-1,0,3], [2,0,-1], [0,2,1]], [[0,0,0], [2,0,-1], [0,-2,1]]])

print(x)
result = []
for c in range(x.shape[0]):
    array = []
    for i in range(2):
        temp = []
        for j in range(2):
            temp.append(np.sum(x[c, i:i+3, j:j+3]*f[c]))
        array.append(temp)
    result.append(array)

a1 = np.sum(np.array(result), axis=0)
print(a1)
a2 = (a1.T + [0, 2]).T
print(a2)









import numpy as np

input_data = np.array([i for j in range(10) for i in [range(120)]]).reshape(10,15,8)
filter = np.array([i for j in range(10) for i in [range(9)]]).reshape(10,3,3)

for i in range(input_data.shape[0]):
    data_pad = np.pad(input_data, pad_width=1, mode='constant', constant_values=0)
    result = []
    for rn in range(data_pad.shape[1] - 2):
        for cn in range(data_pad.shape[2] - 2):
            result.append(np.sum(data_pad[i,rn:rn+3,cn:cn+3] * filter[i]))
    result = np.array(result).reshape(15, 8)

# print(result)





# 문제3
import numpy as np

a = np.array(range(81)).reshape(9,9)

Filter = np.array(range(16)).reshape(4,4)

a_pad = np.pad(a, pad_width=((2,1),(1,2)), mode='constant', constant_values=0)

# 패딩 적용한 합성곱 연산
result = []

for rn in range(len(a_pad)-3):
    for cn in range(len(a_pad)-3):
        result.append(np.sum(a_pad[rn:rn+4, cn:cn+4] * Filter))


print(np.array(result).reshape(9,9))




# 문제4
import numpy as np

a = np.array([i for j in range(3) for i in [range(81)]]).reshape(3,9,9)
filter = np.array([i for j in range(3) for i in [range(16)]]).reshape(3,4,4)

for i in range(a.shape[0]):
    result = []
    a_pad = np.pad(a, pad_width=(1, 2), mode='constant', constant_values=0)
    for rn in range(len(a_pad[1])-3):
        for cn in range(len(a_pad[1])-3):
            result.append(np.sum(a_pad[i,rn:rn+4,cn:cn+4] * filter[i]))
    result = np.array(result).reshape(9,9)

print(result)






# maxpooling 구현
import numpy as np

def pooling(x,pool_h, pool_w):
    result = []
    for rn in range(0, x.shape[0], pool_h):
        for cn in range(0, x.shape[1], pool_w):
            temp = []
            temp.append(x[rn:rn+pool_h, cn:cn+pool_w])
            re = np.max(temp)
            result.append(re)
    result = np.array(result).reshape(2,2)
    print(result)


x = np.array([[21,8,8,12],
              [12,19,9,7],
              [8,10,4,3],
              [18,12,9,10]])

pooling(x,2,2)






# MaxPooling
import numpy as np

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화).

    Parameters
    ----------
    input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩

    Returns
    -------
    col : 2차원 배열
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """(im2col과 반대) 2차원 배열을 입력받아 다수의 이미지 묶음으로 변환한다.

    Parameters
    ----------
    col : 2차원 배열(입력 데이터)
    input_shape : 원래 이미지 데이터의 형상（예：(10, 1, 28, 28)）
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩

    Returns
    -------
    img : 변환된 이미지들
    """
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

data = np.array(
       [[
         [[1, 2, 3, 0],
          [0, 1, 2, 4],
          [1, 0, 4, 2],
          [3, 2, 0, 1]],
         [[3, 0, 6, 5],
          [4, 2, 4, 3],
          [3, 0, 1, 0],
          [2, 3, 3, 1]],
         [[4, 2, 1, 2],
          [0, 1, 0, 4],
          [3, 0, 6, 2],
          [4, 2, 4, 5]]
       ]])

max_pool = Pooling(2, 2)
forward_max = max_pool.forward(data)
print(data.shape)
print(forward_max.shape)
print(data)
print(forward_max)




# Average Pooling
import numpy as np


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화).

    Parameters
    ----------
    input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩

    Returns
    -------
    col : 2차원 배열
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """(im2col과 반대) 2차원 배열을 입력받아 다수의 이미지 묶음으로 변환한다.

    Parameters
    ----------
    col : 2차원 배열(입력 데이터)
    input_shape : 원래 이미지 데이터의 형상（예：(10, 1, 28, 28)）
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩

    Returns
    -------
    img : 변환된 이미지들
    """
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # arg_max = np.argmax(col, axis=1)
        out = np.mean(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        # self.arg_max = arg_max

        return out


data = np.array(
    [[
        [[1, 2, 3, 0],
         [0, 1, 2, 4],
         [1, 0, 4, 2],
         [3, 2, 0, 1]],
        [[3, 0, 6, 5],
         [4, 2, 4, 3],
         [3, 0, 1, 0],
         [2, 3, 3, 1]],
        [[4, 2, 1, 2],
         [0, 1, 0, 4],
         [3, 0, 6, 2],
         [4, 2, 4, 5]]
    ]])

max_pool = Pooling(2, 2)
forward_max = max_pool.forward(data)
print(data.shape)
print(forward_max.shape)
print(data)
print(forward_max)





# 4차원데이터를 생성하고 5x5 가중치로 필터링된 데이터와 합성곱을 하기 편하게 im2col 을 이용해서 2차원으로 변경
import sys, os
import numpy as np
sys.path.append(os.pardir)
from common.util import im2col

x1 = np.random.rand(1,3,7,7)
col1 = im2col(x1,5,5,stride=1,pad=0)
print(col1)




# 0~15 4X4 행렬을 만들고 0~8 3X3 필터를 이용하는데 im2col을 이용
import sys, os
import numpy as np
sys.path.append(os.pardir)
from common.util import im2col

x = np.array(range(16)).reshape(1,1,4,4)
# filter = np.array(range(9)).reshape(1,1,3,3)

col1 = im2col(x,3,3,stride=1, pad=0)

print(col1)




#
import sys, os

sys.path.append(os.pardir)
import numpy as np


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화).

    Parameters
    ----------
    input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩

    Returns
    -------
    col : 2차원 배열
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    for y in range(filter_h):
        y_max = y + stride * out_h  # y=0일 때 y_max=2,  y=1일 때 y_max=3,  y=2일 때 y_max=4
        for x in range(filter_w):
            x_max = x + stride * out_w  # x=0일 때 x_max=2,  x=1일 때 x_max=3,  x=2일 때 x_max=4
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
            # 디버깅 variables화면/ col 우측 클릭 => view as Array 클릭

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    # col.shape = (N,C,filter_h,filter_w,out_h,out_w)  =>  col.shape(N,out_h,out_w,C,filter_h,filter_w) 로 transpose 이후
    # (N*out_h*out_w, C*filter_h*filter_w) 의 2차원 행렬로 reshape
    # print(col.shape)
    return col

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)  # col.shape = (N*out_h*out_w, C*FH*FH)
        # print(col.shape)
        col_W = self.W.reshape(FN, -1).T  # (FN,C,FH,FW)  reshape=> (FN, C*FH*FW)  transpose=> (C*FH*FW, FN)
        out = np.dot(col, col_W) + self.b  # 결과 차원 (N*out_h*out_w,FN)

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        # out.reshape의 shape = (N,out_h,out_w,FN)
        # transpose이후 shape = (N,FN,out_h,out_w)
        return out


x = np.arange(48).reshape(1, 3, 4, 4)
W = np.arange(54).reshape(2, 3, 3, 3)
b = 1
conv = Convolution(W, b)
f = conv.forward(x)
print('f = ', f, 'f.shape = ', f.shape)   # N, FN, out_h, out_w


# 문제1  x1,  W1, b1이 다음과 같을 때 convolution 계층을 거친 뒤 feature map의 차원은?
x1 = np.arange(192).reshape(1, 3, 8, 8)
W1 = np.arange(135).reshape(5, 3, 3, 3)
b1 = 1
conv1 = Convolution(W1, b1)
f1 = conv1.forward(x1)
print('f1 = ', f1, 'f1.shape = ', f1.shape)  # N, FN, out_h, out_w





#
import sys, os

sys.path.append(os.pardir)
import numpy as np


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화).

    Parameters
    ----------
    input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩

    Returns
    -------
    col : 2차원 배열
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    for y in range(filter_h):
        y_max = y + stride * out_h  # y=0일 때 y_max=2,  y=1일 때 y_max=3,  y=2일 때 y_max=4
        for x in range(filter_w):
            x_max = x + stride * out_w  # x=0일 때 x_max=2,  x=1일 때 x_max=3,  x=2일 때 x_max=4
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
            # 디버깅 variables화면/ col 우측 클릭 => view as Array 클릭

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    # col.shape = (N,C,filter_h,filter_w,out_h,out_w)  =>  col.shape(N,out_h,out_w,C,filter_h,filter_w) 로 transpose 이후
    # (N*out_h*out_w, C*filter_h*filter_w) 의 2차원 행렬로 reshape
    # print(col.shape)
    return col

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)  # col.shape = (N*out_h*out_w, C*FH*FH)
        # print(col.shape)
        col_W = self.W.reshape(FN, -1).T  # (FN,C,FH,FW)  reshape=> (FN, C*FH*FW)  transpose=> (C*FH*FW, FN)
        out = np.dot(col, col_W) + self.b  # 결과 차원 (N*out_h*out_w,FN)

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        # out.reshape의 shape = (N,out_h,out_w,FN)
        # transpose이후 shape = (N,FN,out_h,out_w)
        return out


x = np.arange(48).reshape(1, 3, 4, 4)
W = np.arange(54).reshape(2, 3, 3, 3)
b = 1
conv = Convolution(W, b)
f = conv.forward(x)
print('f = ', f, 'f.shape = ', f.shape)   # N, FN, out_h, out_w


# 문제1  x1,  W1, b1이 다음과 같을 때 convolution 계층을 거친 뒤 feature map의 차원은?
x1 = np.arange(192).reshape(1, 3, 8, 8)
W1 = np.arange(135).reshape(5, 3, 3, 3)
b1 = 1
conv1 = Convolution(W1, b1)
f1 = conv1.forward(x1)
print('f1 = ', f1, 'f1.shape = ', f1.shape)  # N, FN, out_h, out_w






# 문제3
import numpy as np
from common.util import im2col

# x = np.random.rand(1,3,7,7)
# col1 = im2col(x,5,5,stride=1, pad=0)
# print(col1)

x = np.array(range(36)).reshape(1,1,6,6)
print(x)
col1 = im2col(x,4,4,stride=1, pad=0)

print(col1.shape)





import numpy as np
from common.util import im2col

x = np.random.rand(1,1,224,224)

col1 = im2col(x,11,11,stride=4, pad=0)
print(col1)












import sys, os

sys.path.append(os.pardir)
import numpy as np


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화).

    Parameters
    ----------
    input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩

    Returns
    -------
    col : 2차원 배열
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    for y in range(filter_h):
        y_max = y + stride * out_h  # y=0일 때 y_max=2,  y=1일 때 y_max=3,  y=2일 때 y_max=4
        for x in range(filter_w):
            x_max = x + stride * out_w  # x=0일 때 x_max=2,  x=1일 때 x_max=3,  x=2일 때 x_max=4
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
            # 디버깅 variables화면/ col 우측 클릭 => view as Array 클릭

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    # col.shape = (N,C,filter_h,filter_w,out_h,out_w)  =>  col.shape(N,out_h,out_w,C,filter_h,filter_w) 로 transpose 이후
    # (N*out_h*out_w, C*filter_h*filter_w) 의 2차원 행렬로 reshape
    # print(col.shape)
    return col

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)  # col.shape = (N*out_h*out_w, C*FH*FH)
        # print(col.shape)
        col_W = self.W.reshape(FN, -1).T  # (FN,C,FH,FW)  reshape=> (FN, C*FH*FW)  transpose=> (C*FH*FW, FN)
        out = np.dot(col, col_W) + self.b  # 결과 차원 (N*out_h*out_w,FN)

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        # out.reshape의 shape = (N,out_h,out_w,FN)
        # transpose이후 shape = (N,FN,out_h,out_w)
        return out


x = np.arange(9).reshape(1, 1, 3, 3)
W = np.arange(4).reshape(1, 1, 2, 2)
b = 0
conv = Convolution(W, b)
f = conv.forward(x)
print('f = ', f)   # N, FN, out_h, out_w




# numpy 를 이용해서 입력데이터를 만들고 변수 x 에 입력
import numpy as np

data = np.array([[
               [[1,2,0,1],[3,0,2,4],[1,0,3,2],[4,2,0,1]],
               [[3,0,4,2],[6,5,4,3],[3,0,2,3],[1,0,3,1]],
               [[4,2,0,1],[1,2,0,4],[3,0,4,2],[6,2,4,5]]]])

print(data.shape)
print(data)




# Pooling 클래스를 생성하고 풀링 결과 출력
import numpy as np

class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad


    def forward(self, x):
        N,C,H,W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (H - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        out = np.max(col, axis=1)

        out = out.reshape(N, out_h, out_w, C).transpose(0,3,1,2)

        return out

data = np.array([[
               [[1,2,3,0],[0,1,2,4],[1,0,4,2],[3,2,0,1]],
               [[3,0,6,5],[4,2,4,3],[3,0,1,0],[2,3,3,1]],
               [[4,2,1,2],[0,1,0,4],[3,0,6,2],[4,2,4,5]]]])



pooling = Pooling(2,2)

print(pooling.forward(data))




# Average Pooling 으로 구현
import numpy as np

class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad


    def forward(self, x):
        N,C,H,W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (H - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        out = np.mean(col, axis=1)

        out = out.reshape(N, out_h, out_w, C).transpose(0,3,1,2)

        return out

data = np.array([[
               [[1,2,3,0],[0,1,2,4],[1,0,4,2],[3,2,0,1]],
               [[3,0,6,5],[4,2,4,3],[3,0,1,0],[2,3,3,1]],
               [[4,2,1,2],[0,1,0,4],[3,0,6,2],[4,2,4,5]]]])



pooling = Pooling(2,2)

print(pooling.forward(data))





# Pooling 클래스에서 backward 추가해서 출력
def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """(im2col과 반대) 2차원 배열을 입력받아 다수의 이미지 묶음으로 변환한다.

    Parameters
    ----------
    col : 2차원 배열(입력 데이터)
    input_shape : 원래 이미지 데이터의 형상（예：(10, 1, 28, 28)）
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩

    Returns
    -------
    img : 변환된 이미지들
    """
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

class Max_Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx

data = np.array([[[[1,2,3,0], [0,1,2,4], [1,0,4,2], [3,2,0,1]],
                  [[3,0,6,5], [4,2,4,3], [3,0,1,0], [2,3,3,1]],
                  [[4,2,1,2], [0,1,0,4], [3,0,6,2], [4,2,4,5]]]])

print(data.shape)

pooling = Max_Pooling(2, 2)
forward_max = pooling.forward(data)
print(forward_max.shape)
backward_max = pooling.backward(forward_max)
print(backward_max.shape)






#
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.trainer import Trainer


class AdaGrad:
    """AdaGrad"""

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class SimpleConvNet:
    """단순한 합성곱 신경망
    conv - relu - pool - affine - relu - affine - softmax
    Parameters
    ----------
    input_size : 입력 크기（MNIST의 경우엔 784）
    hidden_size_list : 각 은닉층의 뉴런 수를 담은 리스트（e.g. [100, 100, 100]）
    output_size : 출력 크기（MNIST의 경우엔 10）
    activation : 활성화 함수 - 'relu' 혹은 'sigmoid'
    weight_init_std : 가중치의 표준편차 지정（e.g. 0.01）
        'relu'나 'he'로 지정하면 'He 초깃값'으로 설정
        'sigmoid'나 'xavier'로 지정하면 'Xavier 초깃값'으로 설정
    """
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)
        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        """손실 함수를 구한다.
        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1: t = np.argmax(t, axis=1)
        acc = 0.0
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)
        return acc / x.shape[0]

    def numerical_gradient(self, x, t):
        """기울기를 구한다（수치미분）.
        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        Returns
        -------
        각 층의 기울기를 담은 사전(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """
        loss_w = lambda w: self.loss(x, t)
        grads = {}
        for idx in (1, 2, 3):
            grads['W' + str(idx)] = numerical_gradient(loss_w, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_w, self.params['b' + str(idx)])
        return grads

    def gradient(self, x, t):
        """기울기를 구한다(오차역전파법).
        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        Returns
        -------
        각 층의 기울기를 담은 사전(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """
        # forward
        self.loss(x, t)
        # backward
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val
        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i + 1)]
            self.layers[key].b = self.params['b' + str(i + 1)]

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
# 시간이 오래 걸릴 경우 데이터를 줄인다.
# x_train, t_train = x_train[:5000], t_train[:5000]
# x_test, t_test = x_test[:1000], t_test[:1000]
max_epochs = 20
network = SimpleConvNet(input_dim=(1, 28, 28),
                        conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)

# 매개변수 보존
network.save_params("params.pkl")
print("Saved Network Parameters!")

# 하이퍼파라미터
iters_num = 10000  # 반복 횟수를 적절히 설정한다.
train_size = x_train.shape[0] # 60000 개
batch_size = 100  # 미니배치 크기
learning_rate = 0.1
train_loss_list = []
train_acc_list = []
test_acc_list = []


adagrad = AdaGrad()

# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)
print(iter_per_epoch) # 600
for i in range(iters_num): # 10000
    # 미니배치 획득  # 랜덤으로 100개씩 뽑아서 10000번을 수행하니까 백만번
    batch_mask = np.random.choice(train_size, batch_size) # 100개 씩 뽑아서 10000번 백만번
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    adagrad.update(network.params, grad)
    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss) # cost 가 점점 줄어드는것을 보려고
    # 1에폭당 정확도 계산 # 여기는 훈련이 아니라 1에폭 되었을때 정확도만 체크
    if i % iter_per_epoch == 0: # 600 번마다 정확도 쌓는다.
        print(x_train.shape) # 60000,784
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc) # 10000/600 개  16개 # 정확도가 점점 올라감
        test_acc_list.append(test_acc)  # 10000/600 개 16개 # 정확도가 점점 올라감
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()






# 특정 필기체 시각화 하는 코드
# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.trainer import Trainer


class SimpleConvNet:
    """단순한 합성곱 신경망

    conv - relu - pool - affine - relu - affine - softmax

    Parameters
    ----------
    input_size : 입력 크기（MNIST의 경우엔 784）
    hidden_size_list : 각 은닉층의 뉴런 수를 담은 리스트（e.g. [100, 100, 100]）
    output_size : 출력 크기（MNIST의 경우엔 10）
    activation : 활성화 함수 - 'relu' 혹은 'sigmoid'
    weight_init_std : 가중치의 표준편차 지정（e.g. 0.01）
        'relu'나 'he'로 지정하면 'He 초깃값'으로 설정
        'sigmoid'나 'xavier'로 지정하면 'Xavier 초깃값'으로 설정
    """

    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))

        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)

        self.params['b1'] = np.zeros(filter_num)

        self.params['W2'] = weight_init_std * \
                            np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """손실 함수를 구한다.

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1: t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]


    def gradient(self, x, t):
        """기울기를 구한다(오차역전파법).

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블

        Returns
        -------
        각 층의 기울기를 담은 사전(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i + 1)]
            self.layers[key].b = self.params['b' + str(i + 1)]


# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 시간이 오래 걸릴 경우 데이터를 줄인다.
# x_train, t_train = x_train[:5000], t_train[:5000]
# x_test, t_test = x_test[:1000], t_test[:1000]


# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 시간이 오래 걸릴 경우 데이터를 줄인다.
# x_train, t_train = x_train[:5000], t_train[:5000]
x_test, t_test = x_test[:10], t_test[:10]
print(t_test)

def filter_show(filters, nx=8, margin=3, scale=10):
    """
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    """
    FN, C, FH, FW = filters.shape
    ny = int(np.ceil(FN / nx))

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(FN):
        ax = fig.add_subplot(ny, nx, i + 1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()


network = SimpleConvNet()
# 무작위(랜덤) 초기화 후의 가중치

# 학습된 가중치
network.load_params("params.pkl")
filter_show(x_test)





# CNN 1층 가중치 시각화
# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.trainer import Trainer


class SimpleConvNet:
    """단순한 합성곱 신경망

    conv - relu - pool - affine - relu - affine - softmax

    Parameters
    ----------
    input_size : 입력 크기（MNIST의 경우엔 784）
    hidden_size_list : 각 은닉층의 뉴런 수를 담은 리스트（e.g. [100, 100, 100]）
    output_size : 출력 크기（MNIST의 경우엔 10）
    activation : 활성화 함수 - 'relu' 혹은 'sigmoid'
    weight_init_std : 가중치의 표준편차 지정（e.g. 0.01）
        'relu'나 'he'로 지정하면 'He 초깃값'으로 설정
        'sigmoid'나 'xavier'로 지정하면 'Xavier 초깃값'으로 설정
    """

    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))

        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)

        self.params['b1'] = np.zeros(filter_num)

        self.params['W2'] = weight_init_std * \
                            np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """손실 함수를 구한다.

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1: t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]


    def gradient(self, x, t):
        """기울기를 구한다(오차역전파법).

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블

        Returns
        -------
        각 층의 기울기를 담은 사전(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i + 1)]
            self.layers[key].b = self.params['b' + str(i + 1)]


# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 시간이 오래 걸릴 경우 데이터를 줄인다.
# x_train, t_train = x_train[:5000], t_train[:5000]
# x_test, t_test = x_test[:1000], t_test[:1000]


# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 시간이 오래 걸릴 경우 데이터를 줄인다.
# x_train, t_train = x_train[:5000], t_train[:5000]
x_test, t_test = x_test[:10], t_test[:10]
# print(t_test)

def filter_show(filters, nx=8, margin=3, scale=10):
    """
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    """
    FN, C, FH, FW = filters.shape
    ny = int(np.ceil(FN / nx))

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(FN):
        ax = fig.add_subplot(ny, nx, i + 1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()


network = SimpleConvNet()
# 무작위(랜덤) 초기화 후의 가중치

# 학습된 가중치
network.load_params("params.pkl")
filter_show(network.params['W1'])
# filter_show(x_test)



def orthogonal(shape):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    return q.reshape(shape)



