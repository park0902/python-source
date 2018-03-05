##############################
#사운드 재생 시간
##############################
import librosa
for i in range(0,17):
    y, sr = librosa.load('D:\park\ccd_sound\\2018-3-3-'+str(i)+'.wav', sr=22050)

    print(i, librosa.get_duration(y=y, sr=sr))




# import pandas as pd
# sound = pd.read_csv("D:\park\sound\\충격음5.csv", header=None)
#
# print(sound.var(axis=1))
# print(sound.var(axis=1)[1])





