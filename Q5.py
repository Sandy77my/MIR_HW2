#!/usr/bin/python
# -*- coding:utf-8 -*-
from glob import glob
import seaborn
import numpy as np , scipy, matplotlib.pyplot as plt
import librosa, librosa.display 
from tqdm import tqdm
import madmom
import mir_eval

import utils

# SMC or JCS
DB = 'JCS'
if DB == 'SMC':
    FILES = glob(DB + '/SMC_MIREX/SMC_MIREX_Audio//*.wav')
elif DB == 'JCS':
    FILES = glob(DB + '/audio//*.wav')

# %% Q5
F_score = list()
sum_f = 0.0
cnt_f = 0.0

for f in tqdm(FILES):
    f = f.replace('\\', '/')
    # print('FILE:', f)

# %%
    # Read the labeled tempo
    g_beats = utils.read_beatfile(DB, f)
    # print('ground-truth beats:\n', g_beats)

    # Beat tracking
    sr, y = utils.read_wav(f)

    # Track beats using time series input
    # Track beats using a pre-computed onset envelope
    onset_env = librosa.onset.onset_strength(y, sr=sr, aggregate=np.median)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env,y=y,sr=sr)
    timetag = librosa.frames_to_time(beats, sr=sr)
    # # print('detect beats:\n', timetag)

    # F score
    f_measure = mir_eval.beat.f_measure(g_beats, timetag, 0.07)
    # print('f_measure:\n', f_measure)
    sum_f += f_measure
    cnt_f += 1.0

F_score = sum_f/cnt_f

print('----------')
print("***** Q5 *****")
print("Database          \tF-score")
# SMC or JCS
print("{:13s}\t{:8.2%}".format('JCS', F_score))

# Plot the onset envelope
hop_length = 512
plt.figure(figsize=(8, 4))
times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
plt.plot(times, librosa.util.normalize(onset_env), label='Onset strength')
plt.vlines(times[beats], 0, 1, alpha=0.5, color='r', linestyle='--', label='Beats')
plt.legend(frameon=True, framealpha=0.75)
# Limit the plot to a 15-second window
plt.xlim(15, 30)
plt.gca().xaxis.set_major_formatter(librosa.display.TimeFormatter())
plt.tight_layout()
plt.show()
