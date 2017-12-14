##############################
#사운드 재생 시간
##############################
# import librosa
# y, sr = librosa.load('D:\park\music\\충격음5.WAV', sr=22050)
#
# print(librosa.get_duration(y=y, sr=sr))




import pandas as pd
sound = pd.read_csv("D:\park\sound\\충격음5.csv", header=None)

print(sound.var(axis=1))
print(sound.var(axis=1)[1])





