#!/usr/bin/python
# -*- coding:utf-8 -*-
from glob import glob
import seaborn
import numpy, scipy, matplotlib.pyplot as plt
import librosa, librosa.display 
from tqdm import tqdm
import madmom

import utils

DB = 'Ballroom'
GENRE = [g.split('/')[2] for g in glob(DB + '/wav/*')]

# %% Q3
genres_P_score, genres_ALOTC_score = list(), list()

for g in tqdm(GENRE):
    # print(GENRE)
    FILES = glob(DB + '/wav/' + g + '/*.wav')
    label, pred_t1, pred_t2, P_score, ALOTC_score = list(), list(), list(), list(), list()

    for f in FILES:
        f = f.replace('\\', '/')
        # print('FILE:', f)

        # Read the labeled tempo
        bpm = float(utils.read_tempofile(DB, f))
        # print(bpm)
        label.append(bpm)

        # madmom estimate tempo
        proc = madmom.features.tempo.TempoEstimationProcessor(fps=100)
        act = madmom.features.beats.RNNBeatProcessor()(f)
        tempo1 = (proc(act)).astype(float)[0][0].item()
        tempo2 = (proc(act)).astype(float)[1][0].item()
        # print(tempo1, tempo2)

        # p score
        s1 = tempo1 / (tempo1 + tempo2)
        s2 = 1.0 - s1
        # print(s1, s2)
        p = s1 * utils.P_score(tempo1, bpm) + s2 * utils.P_score(tempo2, bpm)
        P_score.append(p)

        # ALOTC score
        ALOTC = utils.ALOTC(tempo1, tempo2, bpm)
        ALOTC_score.append(ALOTC)

        # print(p, ALOTC)

    p_avg = sum(P_score) / len(P_score)
    ALOTC_avg = sum(ALOTC_score) / len(ALOTC_score)
    genres_P_score.append(p_avg)
    genres_ALOTC_score.append(ALOTC_avg)

    print('----------')

print(genres_P_score)
print(genres_ALOTC_score)
print()

print("***** Q3 *****")
print("Genre          \tP-score    \tALOTC score")
for g in range(len(GENRE)):
    print("{:13s}\t{:8.2%}\t{:8.2%}".format(GENRE[g], genres_P_score[g], genres_ALOTC_score[g]))
print('----------')
print("Overall P-score:\t{:.2%}".format(sum(genres_P_score) / len(genres_P_score)))
print("Overall ALOTC score:\t{:.2%}".format(sum(genres_ALOTC_score) / len(genres_ALOTC_score)))

# # Plot the onset envelope
# frames = range(len(onset_env))
# t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

# plt.plot(t, onset_env)
# plt.xlim(0, t.max())
# plt.ylim(0)
# plt.xlabel('Time (sec)')
# plt.title('Novelty Function')
# plt.show()

# # Plot the tempogram
# librosa.display.specshow(tempogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='tempo')
# plt.colorbar()
# plt.title('Tempogram')
# plt.tight_layout()
# plt.show()