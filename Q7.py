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
FILES = glob(DB + '/audio//*.wav')

# %% Q7
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

    # madmom beat and downbeat tracker
    proc = madmom.features.downbeats.DBNBarTrackingProcessor(beats_per_bar=[3,4])
    # proc = madmom.features.downbeats.BeatTrackingProcessor(fps=100)
    act = madmom.features.downbeats.RNNBarProcessor()(f)
    timetag = proc(act)

    # F score
    f_measure = mir_eval.beat.f_measure(g_beats, timetag, 0.07)
    print('f_measure:\n', f_measure)
    sum_f += f_measure
    cnt_f += 1.0

F_score = sum_f/cnt_f

print('----------')
print("***** Q7 *****")
print("Database          \tF-score")
# SMC or JCS
print("{:13s}\t{:8.2%}".format('JCS', F_score))