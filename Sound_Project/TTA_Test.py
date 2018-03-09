from librosa import display
import librosa
import librosa.display as dp
import numpy as np
import matplotlib.pyplot as plt



y, sr = librosa.load('D:\park\CCD 소리 수집\긁힘\Scratching-A-Key-On-A-Car-Sound-Effect.wav')
yy, srr = librosa.load('D:\park\CCD 소리 수집\긁힘\scratching-a-key-on-a-car-sound-effect11.wav')
# librosa.output.write_wav('D:\park\CCD 소리 수집\긁힘\Scratching-A-Key-On-A-Car-Sound-Effect1.wav', y, sr)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
mfccs1 = librosa.feature.mfcc(y=yy, sr=srr, n_mfcc=20)
mfccMean = np.mean(mfccs, axis=1)
mfccMean1 = np.mean(mfccs1, axis=1)
print(mfccMean[1])
print(mfccMean1[1])


plt.figure(figsize=(10, 4))
plt.subplot(2,1,1)
# dp.specshow(mfccs, x_axis='time')
dp.waveplot(yy,srr)
# plt.colorbar()
plt.subplot(2,1,2)
dp.waveplot(y,sr)
plt.tight_layout()
plt.show()