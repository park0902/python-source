import glob
import librosa
import numpy as np


def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_files(filenames):
    rows = len(filenames)
    features, labels, groups = np.zeros((rows,193)), np.zeros((rows,8)), np.zeros((rows, 1))
    i = 0
    for fn in filenames:
        try:
            mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            y_col = int(fn.split('/')[3].split('-')[1])
            group = int(fn.split('/')[3].split('-')[0])
        except:
            print(fn)
        else:
            if y_col == 0:
                features[i] = ext_features
                labels[i, 0] = 1
                groups[i] = group
                i += 1
            if y_col == 2:
                features[i] = ext_features
                labels[i, 1] = 1
                groups[i] = group
                i += 1
            if y_col == 3:
                features[i] = ext_features
                labels[i, 2] = 1
                groups[i] = group
                i += 1
            if y_col == 4:
                features[i] = ext_features
                labels[i, 3] = 1
                groups[i] = group
                i += 1
            if y_col == 5:
                features[i] = ext_features
                labels[i, 4] = 1
                groups[i] = group
                i += 1
            if y_col == 7:
                features[i] = ext_features
                labels[i, 5] = 1
                groups[i] = group
                i += 1
            if y_col == 8:
                features[i] = ext_features
                labels[i, 6] = 1
                groups[i] = group
                i += 1
            if y_col == 9:
                features[i] = ext_features
                labels[i, 7] = 1
                groups[i] = group
                i += 1
    return features, labels, groups



audio_files = []
new_audio_files = []

for i in range(1,11):
    audio_files.extend(glob.glob('D:\\park\\002-UrbanSoundData\\audio\\fold%d\*.wav' % i))


for i in range(8732):
    if int(audio_files[i].split('\\')[5].split('-')[1]) == 0 or int(audio_files[i].split('\\')[5].split('-')[1]) == 2 or \
                    int(audio_files[i].split('\\')[5].split('-')[1]) == 3 or int(audio_files[i].split('\\')[5].split('-')[1]) == 4 or \
                    int(audio_files[i].split('\\')[5].split('-')[1]) == 5 or int(audio_files[i].split('\\')[5].split('-')[1]) == 7 or \
                    int(audio_files[i].split('\\')[5].split('-')[1]) == 8 or int(audio_files[i].split('\\')[5].split('-')[1]) == 9:
        new_audio_files.append(audio_files[i])

# print(len(new_audio_files))

print(len(audio_files))






for i in range(9):
    files = new_audio_files[i*1000: (i+1)*1000]
    X, y, groups = parse_audio_files(files)
    for r in y:
        if np.sum(r) > 1.5:
            print('error occured')
            break
    np.savez('D:\\park\\data\\urban_sound_%d' % i, X=X, y=y, groups=groups)



