import librosa
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

knn = KNeighborsClassifier(n_neighbors=1)

sound_data = np.load('D:\park\\urban_sound.npz')
X_data = sound_data['X']
y_data = sound_data['y']

X_sub, X_test, y_sub, y_test = train_test_split(X_data, y_data, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_sub, y_sub, test_size=0.2)

knn.fit(X_train, y_train)

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None,
                     n_jobs=1, n_neighbors=1, p=2, weights='uniform')


def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

mfccs, chroma, mel, contrast, tonnetz = extract_feature('D:\park\\002-UrbanSoundData\\audio\\fold1\\179867-1-0-0.wav')
# mfccs, chroma, mel, contrast, tonnetz = extract_feature('D:\park\\002-UrbanSoundData\\audio\\fold1\\9031-3-4-0.wav')
x_data = np.hstack([mfccs, chroma, mel, contrast, tonnetz]).reshape(1,-1)

print(x_data.shape)

label = {0:'air_conditioner', 1:'car_horn', 2:'children_playing', 3:'dog_bark', 4:'drilling', 5:'engine_idling',
         6:'gun_shot', 7:'jackhammer', 8:'siren', 9:'street_music'}




prediction = knn.predict(x_data)
print(prediction)
print(label[np.argmax(prediction)])

y_pred = knn.predict(X_test)
print(y_pred)

a = np.mean(y_pred == y_test)
print(a)