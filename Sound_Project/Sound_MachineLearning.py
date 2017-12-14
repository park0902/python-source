from librosa import load, stft, feature, get_duration, display as dp
from sklearn import linear_model

y, sr = load('D:\park\music\\긁힘음1.WAV')

mfcc = feature.mfcc(y=y, sr=sr, n_mfcc=20)

print(mfcc, mfcc.shape)

# logreg = linear_model.LogisticRegression()
# logreg.fit(X_train ,y_train)
# y_test_estimated = logreg.predict(X_test)

