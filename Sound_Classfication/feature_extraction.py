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
            print(fn)
        else:
            features[i] = ext_features
            labels[i, y_col] = 1
            groups[i] = group
            i += 1
    return features, labels, groups


audio_files = []
for i in range(1,11):
    audio_files.extend(glob.glob('D:\\park\\002-UrbanSoundData\\audio\\fold%d\*.wav' % i))
print(audio_files)

print(len(audio_files))
for i in range(9):
    files = audio_files[i*1000: (i+1)*1000]
    X, y, groups = parse_audio_files(files)
    for r in y:
        if np.sum(r) > 1.5:
            print('error occured')
            break
    np.savez('D:\\park\\urban_sound_%d' % i, X=X, y=y, groups=groups)





import librosa
import librosa.display as dp
import matplotlib.pyplot as plt
import numpy as np

video_file_path = "/root/test/wma_test.wma"

y, sr = librosa.load(video_file_path)

stft = np.abs(librosa.stft(y))
mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)
tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)

#mfccmean = np.mean(mfcc, axis=1)
print("---mfcc---")
print(mfcc)
print(len(mfcc))

print("---chroma---")
print(chroma)
print(len(chroma))

print("---mel---")
print(mel)
print(len(mel))

print("---contrast---")
print(contrast)
print(len(contrast))

print("---tonnetz---")
print(tonnetz)
print(len(tonnetz))

print(len(mfcc)+len(chroma)+len(mel)+len(contrast)+len(tonnetz))

plt.figure(figsize=(10,4))
dp.waveplot(y=y, sr=sr)
plt.title('avi Wave Graph')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,4))
dp.specshow(mfcc, x_axis='time')
plt.title('avi Spactogram Graph')
plt.colorbar()
plt.tight_layout()
plt.show()


