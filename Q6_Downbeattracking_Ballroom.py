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

DB = 'Ballroom'
GENRE = [g.split('/')[2] for g in glob(DB + '/wav/*')]

# %% Q6
genres_F_score = list()

for g in tqdm(GENRE):
    # print(GENRE)
    FILES = glob(DB + '/wav/' + g + '/*.wav')
    sum_f = 0.0
    cnt_f = 0.0
    
    for f in FILES:
        f = f.replace('\\', '/')
        # print('FILE:', f)

        # Read the labeled tempo
        g_beats = utils.read_beatfile(DB, f)
        # print('ground-truth beats:\n', g_beats)

        # Beat tracking
        sr, y = utils.read_wav(f)
        
        # madmom beat tracking
        proc = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
        act = madmom.features.downbeats.RNNDownBeatProcessor()(f)
        timetag = proc(act)
        np_time = timetag[:, 0]

        # F score
        f_measure = mir_eval.beat.f_measure(g_beats, np_time, 0.07)
        # print('f_measure:\n', f_measure)
        sum_f += f_measure
        cnt_f += 1.0

    genres_F_score.append(sum_f/cnt_f)

print('----------')
print(genres_F_score)
print()

print("***** Q6 *****")
print("Genre          \tF-score")
for g in range(len(GENRE)):
    print("{:13s}\t{:8.2%}".format(GENRE[g], genres_F_score[g]))
print('----------')
print("Overall F-score:\t{:.2%}".format(sum(genres_F_score)/len(genres_F_score)))