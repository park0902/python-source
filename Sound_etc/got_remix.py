import math
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load audio from file
video_file_path = "D:\park\music\\Shock2.WAV"
audio_data, sr = librosa.load(video_file_path)

max_limit = 0.005
frame_size = 0.1
frame_len = frame_size * sr

# Calculate the number of frames in the audio file
num_of_frames = math.floor(len(audio_data) / frame_len)

silent_frames_indexes = []
for frame_num in range(int(num_of_frames)):
    # Get the start and end of the frame
    start = int(frame_num * frame_len)
    stop = int((frame_num + 1) * frame_len)

    # Get the absolute values in each frame
    abs_frame = map(abs, audio_data[start:stop])

    # Get the maximum value in the frame
    cur_max_val = max(abs_frame)

    # We've reached a "silent" frame if the maximum
    # value in the frame below our limit
    if cur_max_val < max_limit:
        silent_frames_indexes.append(frame_num)

print(silent_frames_indexes)

plt.figure(figsize=(10, 4))
librosa.display.waveplot(audio_data)
plt.plot([0.002] * len(audio_data))
plt.plot([-0.002] * len(audio_data))
# librosa.display.specshow(silent_frames_indexes, x_axis='time')
# plt.colorbar()
# plt.title('{} Mfcc Graph'.format("Shock1"))
# plt.tight_layout()

plt.show()