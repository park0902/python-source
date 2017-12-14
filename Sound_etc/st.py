from librosa import load, stft, feature, get_duration, display as dp
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
import os

################## 플레이 정보 ##################
play_duration = 3                # 노래 재생 시간
file_loc = 'D:\park\music' # 노래 폴더 위치
show_chroma = True              # 크로마그램 출력
###############################################


################## 사운드 클래스 ##################
class Sound(object):
    def __init__(self,show=show_chroma):
        self.music_dict = {}  # 사운드 이름을 key로, 파일경로를 value로 하는 딕셔너리
        self.music = ''       # 재생할 노래 이름
        self.time = 0         # 노래 총 길이
        self.show = show      # 크로마그램 화면에 출력할지 말지 여부
        self.chroma_len = 0

    ##### 사운드 로딩 메소드 #####

    # (사운드 로딩 Main) 노래 로드 메인 메소드(여기서 리턴한 노래의 코사인 유사도 정보를 Analysis() 메소드에서 분석)
    def _LoadSong(self):
        self.music_dict = self._SearchSong(file_loc)  # 사운드, 파일경로 딕셔너리 생성
        self.music = self._Input()                    # 재생할 사운드 이름 입력

        y, sr = load(self.music_dict[self.music], sr=880, duration=3) # librosa 모듈의 노래 불러오기(샘플링은 0.5초 당 하나씩)

        self.mfccs = feature.mfcc(y=y,sr=880).T

        s = np.abs(stft(y)**2)                            # 노래의 파워 데이터(주파수의 진폭)
        self.time = get_duration(y=y, sr=sr)              # 노래의 총 길이(샘플링 데이터 갯수)
        chroma = feature.chroma_stft(S=s, sr=sr)          # 크로마그램으로 변환
        self._Chromagram(chroma, show=self.show, title='music chromagram_line41')  # 크로마그램 그래프를 출력
        # chromaT = np.transpose(chroma,axes=(1,0))         # time-time 코사인 유사도 구하기 위해 전치행렬로 변환
        # self._Chromagram(chromaT, show=self.show, title='music transpose_line43')  # 전치화된 크로마그램 그래프를 출력
        print('\nLoading Finished!')
        print(y.shape, self.mfccs, self.mfccs.shape)
        # print(mfccs.shape)
        print(y.shape, sr)
        # return cosine_similarity(self.mfccs).shape                 # 사운드 각 부분의 코사인 유사도를 리턴


    # (사운드 로딩 SUB) 폴더에서 WAV 파일 리스트를 뽑아서 딕셔너리에 담는 메소드
    def _SearchSong(self, dirname):
        filenames = os.listdir(dirname)                     # 지정된 폴더 내 파일이름들 불러오기
        sound_dict = {}
        for filename in filenames:
            full_filename = os.path.join(dirname, filename) # full_filename = 경로+파일이름
            ext = os.path.splitext(full_filename)[-1]       # ext에 확장자 넣기
            file = os.path.splitext(filename)[0]            # file에 확장자를 제외한 파일이름만 넣기
            if ext == '.WAV':                               # 확장자가 WAV 인 파일만 sound_dict 딕셔너리에 넣기
                sound_dict[file] = full_filename            # 파일이름(key), 경로+파일이름(value)
        return sound_dict                                   # sound_dict 딕셔너리 리턴

    # (사운드 로딩 SUB) 처음에 사운드 리스트를 화면에 출력하고 재생할 사운드 입력받는 메소드
    def _Input(self):
        print('---------------------------------------------------'
              'Sound List---------------------------------------------------')
        sound_dict_list = list(self.music_dict.keys())
        for idx in range(len(sound_dict_list)//5 + 1):
            try:
                print(' '.join([i.ljust(25) for i in sound_dict_list[5 * idx : 5 * idx +5]]))
            except IndexError:
                print(' '.join([i.ljust(25) for i in sound_dict_list[5 * idx: -1]]))

        return input('\n원하는 사운드 제목을 입력하세요. ')


    def _Chromagram(self, chroma, show=False, title=None):
        if show == True:
            plt.figure(figsize=(10, 4))
            dp.specshow(chroma, x_axis='time')
            plt.colorbar()
            # plt.title('{}'.format(self.music.upper()))
            plt.title(title)
            plt.tight_layout()
            plt.show()


################## 메인 실행절 ##################
if __name__ == '__main__':
    sound = Sound()
    sound._LoadSong()





