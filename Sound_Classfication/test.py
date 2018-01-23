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
    features, labels, groups = np.zeros((rows,193)), np.zeros((rows,10)), np.zeros((rows, 1))
    i = 0
    for fn in filenames:
        try:
            mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            y_col = int(fn.split('/')[3].split('-')[1])
            group = int(fn.split('/')[3].split('-')[0])
        except:
            # print(fn)
            pass
        else:
            features[i] = ext_features
            labels[i, y_col] = 1
            groups[i] = group
            i += 1
    print(rows, features, labels, groups)

# print(extract_feature("D:\park\\002-UrbanSoundData\\audio\\fold1\\7061-6-0-0.wav"))
# parse_audio_files("D:\park\\002-UrbanSoundData\\audio\\fold1\\7061-6-0-0.wav")

print(parse_audio_files("D:\\park\\002-UrbanSoundData\\audio\\fold1\*.wav"))




fn = "UrbanSound8K/audio/fold%d/7061-6-0-0.wav"
y_col = int(fn.split('/')[3].split('-')[1])
group = int(fn.split('/')[3].split('-')[0])

print(y_col)
print(group)