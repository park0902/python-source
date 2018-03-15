import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display as dp

#####################################################################
# matplotlib 에서 한글 깨짐 방지 코드
#####################################################################
from matplotlib import font_manager, rc
font_fname = 'c:/windows/fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname=font_fname).get_name()
rc('font', family=font_name)
#####################################################################

time_list = []
for i in range(1,26):
    y, sr = librosa.load("D:\녹음차량 sample20180214\\충격 sample "+str(i)+".WAV")
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccMean = np.mean(mfccs, axis=1)
    #
    # plt.figure(figsize=(10, 4))
    # dp.specshow(mfccs, x_axis='time')
    # plt.colorbar()
    # plt.title('긁힘 sample {} Mfcc Graph'.format(i))
    # plt.tight_layout()
    # plt.savefig('D:\CCD_MFCC\\충격 sample'+str(i)+'_mfcc'+ '.png')
    # # plt.show()
    #
    # plt.figure(figsize=(10, 4))
    # dp.waveplot(y, sr=sr)
    # plt.title('긁힘 sample {} Mfcc Graph'.format(i))
    # plt.tight_layout()
    # plt.savefig('D:\CCD_MFCC\\충격 sample'+str(i)+'_wave'+ '.png')
    # # plt.show()
    #
    #
    # mfccMeans = np.mean(mfccs, axis=1)
    # np.savetxt("D:\CCD_MFCC\\충격 sample"+str(i)+"_mfccMean.csv", mfccMeans, delimiter=",")


    time = librosa.get_duration(y=y, sr=sr)
    time_list.append(time)

np.savetxt("D:\CCD_MFCC\\충격 sample_time.csv", time_list, delimiter=",")