import librosa
import numpy as np
import sys

file_path = sys.argv[1]

y, sr = librosa.load(file_path)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=100)
mfccMean = np.mean(mfccs, axis=1)

if 80 <= mfccMean[1] <= 130:
    print("1")
else:
    print("4")





