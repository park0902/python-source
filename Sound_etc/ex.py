# filename = "64f999a2b468daf4_2490_2520.wav"
#  y, sr = librosa.load(filename)
#
# Calculate mfccs.
#
# Y = librosa.stft(y)
#  mfccs = librosa.feature.mfcc(y)
#
# Build reconstruction mappings,
#
# n_mfcc = mfccs.shape[0]
#  n_mel = 128
#  dctm = librosa.filters.dct(n_mfcc, n_mel)
#  n_fft = 2048
#  mel_basis = librosa.filters.mel(sr, n_fft)
#
# Empirical scaling of channels to get ~flat amplitude mapping.
#
# bin_scaling = 1.0/np.maximum(0.0005, np.sum(np.dot(mel_basis.T, mel_basis),
#  axis=0))
#
# Reconstruct the approximate STFT squared-magnitude from the MFCCs.
#
# recon_stft = bin_scaling[:, np.newaxis] * np.dot(mel_basis.T,
#  invlogamplitude(np.dot(dctm.T, mfccs)))

# Impose reconstructed magnitude on white noise STFT.
#
# excitation = np.random.randn(y.shape[0])
#  E = librosa.stft(excitation)
#  recon = librosa.istft(E/np.abs(E)*np.sqrt(recon_stft))


########################################################################################################################
# The audio information:
#  Input File : 'aa.wav'
#  Channels : 1
#  Sample Rate : 16000
#  Precision : 16-bit
#  Duration : 00:00:00.64 = 10160 samples ~ 47.625 CDDA sectors
#  File Size : 20.4k
#  Bit Rate : 257k
#  Sample Encoding: 16-bit Signed Integer PCM
#
# when use the "frame length=25ms, frame shift=10ms" , number of frames should be
#  (10160-240)/160=62frames. and get 62 frame in kaldi.
#
# But use librosa to extract the MFCC features, I got 64 frames:
#
# sr = 16000
#  n_mfcc = 13
#  n_mels = 40
#  n_fft = 512
#  win_length = 400 # 0.025*16000
#  hop_length = 160 # 0.010 * 16000
#  window = 'hamming'
#  fmin = 20
#  fmax = 4000
#  y, sr = librosa.load(wav_file, sr=16000)
#  print(sr)
#  D = numpy.abs(librosa.stft(y, window=window, n_fft=n_fft, win_length=win_length, hop_length=hop_length))**2
#  S = feature.melspectrogram(S=D, y=y, n_mels=n_mels, fmin=fmin, fmax=fmax)
#  feats = feature.mfcc(S=librosa.power_to_db(S), n_mfcc=n_mfcc)
#  print(feats.shape)
#
# OR
#
# feats = feature.mfcc(y=y, n_mfcc=n_mfcc, n_fft=n_fft, n_mels=n_mels, fmin=fmin, hop_length=hop_length)
#
# all of the two librosa code will result in (13,64) shape.
#
# Another question, in the feature.mfcc() function:
#  Could I directly given the window_length, window, hop_length parameters?
#
# Look forward to your reply.
#
# Thanks
#  Jinming

########################################################################################################################

# from librosa import load, display, core, stft, feature, core
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# y, sr = load("D:\park\music\\Shock1.WAV", sr=22050)
# # Tgram = feature.tempogram(y=y, sr=sr)
# D = core.amplitude_to_db(stft(y), ref=np.max)
# # chroma = feature.chroma_stft(y=y, sr=sr)
# plt.figure(figsize=(10, 4))
# display.specshow(D, x_axis='time')
# plt.colorbar()
# plt.title('Tempogram')
# plt.tight_layout()
#

# import numpy as np
# ts = [np.array([[1,2,3],[2,3,4],[4,3,2]])]

# print(np.mean(ts, axis=1))
#
# for i in range(2):
#     print(np.mean(ts[i], axis=1))
    # np.savetxt("D:\park\sound\\" +"Sractch_Mean"+ "_mfcc.csv", mfccMean, delimiter=",")

# d = np.abs(stft(y))
#
# print(a)




# from librosa import load, display, feature, effects
# import matplotlib.pyplot as plt
#
# y, sr = load("D:\park\sound\\Scratch1.WAV", sr=22050) # librosa 모듈의 노래 불러오기(샘플링 주파수 22050Hz))
# # y : 오디오 시계열 데이터, sr : y의 샘플링 비율
#
# y_harm, y_perc = effects.hpss(y)
#
# print(y_harm, y_perc)
#
#
# mfccs = feature.mfcc(y=y,sr=sr, n_mfcc=20)
# # mfcc의 계수인 n_mfcc=20으로 설정(보통 20~50)
#
# plt.figure(figsize=(10, 4))
# display.specshow(mfccs, x_axis='time')    # mfcc graph 생성
# plt.show()



# from librosa import load, util, display, core, stft, feature
import numpy as np
import matplotlib.pyplot as plt
import librosa



y, sr = librosa.load("D:\park\music\\rainbow.mp3")

print(y, sr)


