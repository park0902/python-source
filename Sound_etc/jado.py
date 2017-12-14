import glob
import librosa
import numpy as np
import librosa.display

yt = []
srt = []
# for i in range(7):
#     file = 'D:\park\music\\Scratch'+str(i + 1) + '.WAV'
#
#     yt, srt = librosa.load(file)
#     y.append(yt)
#     sr.append(srt)


# import numpy as np
# import librosa.display
#
# y, sr = librosa.load('D:\park\music\\Scratch1.WAV')
#
#
# mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
# mfccMean = np.mean(mfcc, axis=1)
# print(mfccMean)



# for i in range(7):
    # print(mfccMean[i][1])
    # np.savetxt("D:\park\sound\\" +"Sractch_Mean"+ "_mfcc.csv", mfccMean[i][1], delimiter=",")



for i in range(7):
    file = '\\root\\test\\Scratch'+str(i + 1) + '.WAV'
    y, sr = librosa.load(file)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccmean = np.mean(mfcc, axis=1)
    print(mfccmean)