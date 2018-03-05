import glob
import numpy as np

X = np.empty((0, 193))
y = np.empty((0, 4))
groups = np.empty((0, 1))
npz_files = glob.glob('D:\park\ccd_sound_data\\ccd_sound_data.npz')

print(npz_files)



for fn in npz_files:
    print(fn)
    data = np.load(fn)
    X = np.append(X, data['X'], axis=0)
    y = np.append(y, data['y'], axis=0)
    groups = np.append(groups, data['groups'], axis=0)

print(groups[groups>0])

print(X.shape, y.shape)
for r in y:
    if np.sum(r) > 1.5:
        print(r)
np.savez('D:\park\ccd_sound_data\\ccd_sound', X=X, y=y, groups=groups)