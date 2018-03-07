##################################################################################
#사운드 재생 시간
##################################################################################
# import librosa
# for i in range(0,17):
#     y, sr = librosa.load('D:\park\ccd_sound\\2018-3-3-'+str(i)+'.wav', sr=22050)
#
#     print(i, librosa.get_duration(y=y, sr=sr))
##################################################################################

from librosa import display
import librosa
import numpy as np

# for i in range(0,7):
#     y, sr = librosa.load('D:\park\\new_sound\\2018-0-3-'+str(i)+'.wav', sr=22050)
#     print(i, librosa.get_duration(y=y, sr=sr))


# for i in range(0,1):
#     y, sr = librosa.load('D:\park\\new_sound\\2018-0-3-'+str(i)+'.wav')
#     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=100)
#     wave = display.waveplot(y, sr)
#     print(wave)


for i in range(0,1):
    y, sr = librosa.load('D:\park\\new_sound\\2018-0-3-'+str(i)+'.wav')
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=100)
    mfccMean = np.mean(mfccs, axis=1)
    # chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    # chroma_med = librosa.decompose.nn_filter(chroma, aggregate=np.median, metric='cosine')
    print(mfccMean[1])
    # print(chroma.shape)
    # print(chroma_med)
    # print(chroma_med.shape)
    # np.savetxt("D:\park\\new_sound\\"+"2018-0-3-"+str(i)+"_mfccMean(100).csv", mfccMean, delimiter=",")



# for i in range(0,19):
#     y, sr = librosa.load('D:\park\ccd_sound\\2018-0-3-'+str(i)+'.wav')
#     stft = np.abs(librosa.stft(y))
#     chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
#     print(i, chroma)
#
# for i in range(0,19):
#     y, sr = librosa.load('D:\park\ccd_sound\\2018-0-3-'+str(i)+'.wav')
#     mel = np.mean(librosa.feature.melspectrogram(y, sr=sr).T, axis=0)
#     print(i, mel)
#
# for i in range(0,19):
#     y, sr = librosa.load('D:\park\ccd_sound\\2018-0-3-'+str(i)+'.wav')
#     stft = np.abs(librosa.stft(y))
#     contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)
#     print(i, contrast)
#
# for i in range(0,19):
#     y, sr = librosa.load('D:\park\ccd_sound\\2018-0-3-'+str(i)+'.wav')
#     tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
#     print(i, tonnetz)







# import pandas as pd
# sound = pd.read_csv("D:\park\sound\\충격음5.csv", header=None)
#
# print(sound.var(axis=1))
# print(sound.var(axis=1)[1])





