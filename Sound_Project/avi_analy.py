import librosa
import librosa.display as dp
import matplotlib.pyplot as plt
import numpy as np

# Load audio from file
video_file_path = "D:\park\\002-UrbanSoundData\\audio\\fold1\\57320-0-0-7.wav"
y, sr = librosa.load(video_file_path)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

print(mfcc)

plt.figure(figsize=(10, 4))
dp.waveplot(y, sr=sr)
# plt.plot([0.005] * len(y))
# plt.plot([-0.005] * len(y))

plt.tight_layout()
# plt.savefig('D:\park\sound\\'+self.music+'_wave'+'.png')
plt.show()
# Display wave plot
# librosa.display.waveplot(audio_data)
#
# # Add limits between -0.02 and 0.02
# plt.plot([0.02] * len(audio_data))
# plt.plot([-0.02] * len(audio_data))
#
# # Show the plot
# plt.show()



