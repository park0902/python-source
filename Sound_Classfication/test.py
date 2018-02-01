# import glob
# import librosa
# import numpy as np
#
#
# def extract_feature(file_name):
#     X, sample_rate = librosa.load(file_name)
#     stft = np.abs(librosa.stft(X))
#     mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
#     chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
#     mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
#     contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
#     tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
#     return mfccs,chroma,mel,contrast,tonnetz
#
# def parse_audio_files(filenames):
#     rows = len(filenames)
#     features, labels, groups = np.zeros((rows,193)), np.zeros((rows,10)), np.zeros((rows, 1))
#     i = 0
#     for fn in filenames:
#         try:
#             mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
#             ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
#             y_col = int(fn.split('/')[3].split('-')[1])
#             group = int(fn.split('/')[3].split('-')[0])
#         except:
#             # print(fn)
#             pass
#         else:
#             features[i] = ext_features
#             labels[i, y_col] = 1
#             groups[i] = group
#             i += 1
#     print(rows, features, labels, groups)
#
# # print(extract_feature("D:\park\\002-UrbanSoundData\\audio\\fold1\\7061-6-0-0.wav"))
# # parse_audio_files("D:\park\\002-UrbanSoundData\\audio\\fold1\\7061-6-0-0.wav")
#
# print(parse_audio_files("D:\\park\\002-UrbanSoundData\\audio\\fold1\*.wav"))
#
#
#
#
# fn = "UrbanSound8K/audio/fold%d/7061-6-0-0.wav"
# y_col = int(fn.split('/')[3].split('-')[1])
# group = int(fn.split('/')[3].split('-')[0])
#
# print(y_col)
# print(group)

import numpy as np
# 파일 로드(5가지 Sound 음성 패턴 추출한.npz 파일)
sound_data = np.load('D:\\park\\urban_sound.npz')

# 데이터 추출(X_data : 음성 추출 데이터, y_data : 라벨(0~9))
X_data = sound_data['X']
y_data = sound_data['y']
new_y_data = []


for i in range(len(y_data)):
    if (np.argmax(y_data[i]) == 0) or (np.argmax(y_data[i]) ==2) or \
            (np.argmax(y_data[i]) == 3) or (np.argmax(y_data[i]) ==4) or\
            (np.argmax(y_data[i]) == 5) or (np.argmax(y_data[i]) ==7) or\
            (np.argmax(y_data[i]) == 8) or (np.argmax(y_data[i]) ==9):
        new_y_data.append(y_data[i])
new_y_data = np.array(new_y_data)



# for i in range(len(y_data)):
#     if np.argmax(y_data[i]) == 6:
#         b.append(i)
#
# print(len(b))
# print(b)



# print(len(new_y_data))