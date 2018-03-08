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
import librosa.display as dp
import numpy as np
import matplotlib.pyplot as plt

# for i in range(0,7):
#     y, sr = librosa.load('D:\park\\new_sound\\2018-0-3-'+str(i)+'.wav', sr=22050)
#     print(i, librosa.get_duration(y=y, sr=sr))


# for i in range(0,1):
#     y, sr = librosa.load('D:\park\\new_sound\\2018-0-3-'+str(i)+'.wav')
#     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=100)
#     wave = display.waveplot(y, sr)
#     print(wave)

plt.figure(figsize=(10, 4))
for i in range(0, 3):
    y, sr = librosa.load('D:\park\\new_sound\\2018-0-3-'+str(i)+'.wav')
    # mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=int(sr * 0.01), n_fft=int(sr * 0.02),n_mfcc=20).T
    # mfcc_delta = librosa.feature.delta(mfccs)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccMean = np.mean(mfccs, axis=1)
    print(mfccMean[1])
    # plt.figure(figsize=(10, 4))
    plt.subplot(3,1,i+1)
    dp.specshow(mfccs, x_axis='time')
# plt.colorbar()
# plt.subplot(2,1,2)
# dp.specshow(mfcc_delta, x_axis='time')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

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





