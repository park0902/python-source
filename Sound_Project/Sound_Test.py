from librosa import load, feature, core, stft, display as dp
import matplotlib.pyplot as plt
import numpy as np
import os
import librosa
librosa.util.example_audio_file()
################## 플레이 정보 ##################
file_loc = 'D:\park\ccd_sound'      # 노래 폴더 위치
show_graph = True              # 그래프 출력
###############################################


################## 사운드 클래스 ##################
class Sound(object):
    def __init__(self,show=show_graph):
        self.music_dict = {}  # 사운드 이름을 key로, 파일경로를 value로 하는 딕셔너리
        self.music = ''       # 재생할 노래 이름
        self.time = 0         # 노래 총 길이
        self.show = show      # 크로마그램 화면에 출력할지 말지 여부
        self.chroma_len = 0

        self.mfcc = []
        self.mfccMean = []
        self.mfccMean_test = []

    ##### 사운드 로딩 메소드 #####
    # (사운드 로딩 Main)
    def _LoadSong(self):
        self.music_dict = self._SearchSong(file_loc)  # 사운드, 파일경로 딕셔너리 생성
        self.music = self._Input()                    # 재생할 사운드 이름 입력

        y, sr = load(self.music_dict[self.music], sr=22050) # librosa 모듈의 노래 불러오기(샘플링 주파수 22050Hz)
        d = np.abs(stft(y))

        self.chroma = feature.chroma_stft(y=y, sr=sr)
        self.mfccs = feature.mfcc(y=y,sr=sr, n_mfcc=20)

        # self.mfcc.append(feature.mfcc(y=y, sr=sr, n_mfcc=20))

        self._Mfcc(self.mfccs, show=self.show)
        self._Wave(y,sr, show=self.show)
        self._Spactogram(d, show=self.show)
        self._Chromagram(self.chroma, show=self.show)

        print('\nLoading Finished!')
        print(self.mfccs, self.mfccs.shape)


    # (사운드 로딩 SUB) 폴더에서 WAV 파일 리스트를 뽑아서 딕셔너리에 담는 메소드
    def _SearchSong(self, dirname):
        filenames = os.listdir(dirname)                     # 지정된 폴더 내 파일이름들 불러오기
        sound_dict = {}
        for filename in filenames:
            full_filename = os.path.join(dirname, filename) # full_filename = 경로+파일이름
            ext = os.path.splitext(full_filename)[-1]       # ext에 확장자 넣기
            file = os.path.splitext(filename)[0]            # file에 확장자를 제외한 파일이름만 넣기
            if ext == '.wav':                               # 확장자가 WAV 인 파일만 sound_dict 딕셔너리에 넣기
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


    #
    def _Mfcc(self, mfcc, show=False):
        if show == True:
            plt.figure(figsize=(10, 4))
            dp.specshow(mfcc, x_axis='time')
            plt.colorbar()
            plt.title('{} Mfcc Graph'.format(self.music))
            plt.tight_layout()
            plt.savefig('D:\park\ccd_sound_data\\'+self.music+'_mfcc'+ '.png')
            # plt.show()

    def _Wave(self, y, sr, show=False):
        if show == True:
            plt.figure(figsize=(10, 4))
            dp.waveplot(y, sr=sr)
            # plt.plot([0.005] * len(y))
            # plt.plot([-0.005] * len(y))
            plt.title('{} Wave Graph'.format(self.music))
            plt.tight_layout()
            plt.savefig('D:\park\ccd_sound_data\\'+self.music+'_wave'+'.png')
            # plt.show()

    def _Spactogram(self, d, show=False):
        if show == True:
            plt.figure(figsize=(10, 4))
            dp.specshow(core.amplitude_to_db(d), x_axis='time')
            plt.colorbar()
            plt.title('{} Spactogram Graph'.format(self.music))
            plt.tight_layout()
            plt.savefig('D:\park\ccd_sound_data\\'+self.music+'_spactogram'+ '.png')
            # plt.show()

    def _Chromagram(self, chroma, show=False):
        if show == True:
            plt.figure(figsize=(10, 4))
            dp.specshow(chroma, x_axis='time')
            plt.colorbar()
            plt.title('{} Chromagram Graph'.format(self.music))
            plt.tight_layout()
            plt.savefig('D:\park\ccd_sound_data\\'+self.music+'_Chromagram'+ '.png')
            # plt.show()


    def _Save(self):
        np.savetxt("D:\park\ccd_sound_data\\"+self.music+"_mfcc.csv", self.mfccs, delimiter=",")



################# 메인 실행절 ########################
# if __name__ == '__main__':
#     sound = Sound()
#     sound._LoadSong()
####################################################


################## 메인 저장 실행절 ##################
if __name__ == '__main__':
    cnt = 0
    while True:
        sound = Sound()
        sound._LoadSong()
        sound._Save()
        cnt += 1
        if cnt == len(sound.music_dict):
            break
####################################################



# import librosa
# import librosa.display as dp
# import matplotlib.pyplot as plt
# import numpy as np
#
# video_file_path = '/root/test/mp4_test.mp4'
# y, sr= librosa.load(video_file_path)
#
# mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
# mfccmean = np.mean(mfcc, axis=1)
#
# print("---mfcc---")
# print(mfcc)
#
# print("--- mfcc mean ---")
# print(mfccmean)
#
# plt.figure(figsize=(10,4))
# dp.waveplot(y=y, sr=sr)
# plt.title('mp4 Wave Graph')
# plt.show()
#
# plt.figure(figsize=(10,4))
# dp.specshow(mfcc, x_axis='time')
# plt.colorbar()
# plt.title('mp4 Spactogram Graph')
# plt.show()
#


