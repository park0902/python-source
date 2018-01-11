# import pydub
# sound = pydub.AudioSegment.from_mp3("D:/park/music/sample.mp3")
# sound.export("D:/park/music/apple.wav", format="wav")

# import os
# import pydub
# import glob
#
# wav_files = glob.glob('D:\park\music\\*.WAV')
# print(wav_files)
#
# for wav_file in wav_files:
#     mp3_file = os.path.splitext(wav_file)[0] +'.mp3'
#     sound = pydub.AudioSegment.from_wav(wav_file)
#     sound.export(mp3_file, format='mp3')
#     os.remove(wav_file)

# from pydub import AudioSegment
# # Open file
#
# song = AudioSegment.from_wav('D:/park/music/Shock.WAV')
#
# print(song)

import ffmpy
ff = ffmpy.FFmpeg(
    inputs={'D:/park/music/sample.mp3': None},
    outputs={'D:/park/music/sample.wav': None})
ff.run()



import subprocess


