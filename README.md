# MIR_HW2

## Overview
In this homework we will implement algorithms for the following tasks: 
1. Compute the tempo of a song
2. Identify every beat / downbeat position of a song
3. Identify the meters of a song

## Requirements
- Download the Ballroom dataset and annotation from [https://drive.google.com/open?id=1Gk81pTyo65FIkUdR3inlNVWm72cqeaII](https://drive.google.com/open?id=1Gk81pTyo65FIkUdR3inlNVWm72cqeaII)
- Download the SMC beat tracking dataset and annotation [http://smc.inesctec.pt/research/data-2/](http://smc.inesctec.pt/research/data-2/)
- Download the JCS beat and downbeat tracking dataset and annotation: [https://drive.google.com/drive/folders/18OP9LU8YflZtkULOk7qLAZkdBY8cOQfn](https://drive.google.com/drive/folders/18OP9LU8YflZtkULOk7qLAZkdBY8cOQfn)
- Download the madmom library, the state-of-the-art Python library for beat and downbeat tracking: [https://madmom.readthedocs.io/en/latest/](https://madmom.readthedocs.io/en/latest/)
- There are also Python resources available for tempo and beat computation: [https://bmcfee.github.io/librosa/generated/librosa.feature.tempogram.html](https://bmcfee.github.io/librosa/generated/librosa.feature.tempogram.html) [https://librosa.github.io/librosa/generated/librosa.beat.tempo.html](https://librosa.github.io/librosa/generated/librosa.beat.tempo.html) [https://librosa.github.io/librosa/generated/librosa.beat.beat_track.html](https://librosa.github.io/librosa/generated/librosa.beat.beat_track.html)

## Environment
- Ubuntu 18.04 LTS
- Python 3

## Task 1 - Tempo Estimation

## Q1

### Problem Statement

- Design an algorithm that estimate the tempo of each clip in the Ballroom dataset. Assume that the tempo of every clip is constant. Note that your algorithm should output two predominant tempi for each clip: T1 (the slower one) and T2 (the faster one).
- For example, you may simply try the two largest peak values in the tempogram of the clip. The tempogram can be computed from **librosa.feature.tempogram**. The evaluation method of tempo estimation is as follows.
- Compute the average P-scores and the ALOTC scores of the eight genres (Cha Cha, Jive, Quickstep, Rumba, Samba, Tango, Viennese Waltz and Slow Waltz) in the Ballroom dataset using your algorithm. The above process can all be found in the evaluation routine **mir_eval.tempo.detection**.

### Results
![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5c5710d4-0cbd-4aba-988f-9162ac2be7bb/Q1_512_tempogram.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5c5710d4-0cbd-4aba-988f-9162ac2be7bb/Q1_512_tempogram.png)

***Hop length=512***
![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bb7225a2-2a22-4e49-9724-c32d082b17ae/Q1_1024_tempogram.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bb7225a2-2a22-4e49-9724-c32d082b17ae/Q1_1024_tempogram.png)

***Hop length=1024***

## Q2

### Problem Statement

- Instead of using your estimated [T1 ,T2 ] in evaluation, try to use [T1 /2,T2 /2], [T1 /3,T2 / 3], [2T1 ,2T2 ], and [3T1 ,3T2 ] for estimation. What are the resulting P-scores? Discuss the results.

### Results
***P-score & ALOTC score of 8 Genres. [T1 /2,T2 /2]***
![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5f34e45b-be58-4016-b90d-725e11270c07/Q2_d2.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5f34e45b-be58-4016-b90d-725e11270c07/Q2_d2.png)
![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a54b1ba8-fcf0-480b-97c2-9943785959d0/Q2_d2_tempogram.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a54b1ba8-fcf0-480b-97c2-9943785959d0/Q2_d2_tempogram.png)

***P-score & ALOTC score of 8 Genres. [T1 /3,T2 /3]***
![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/01cba9d0-7ea6-4696-8378-3acbffd4f7e2/Q2_d3.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/01cba9d0-7ea6-4696-8378-3acbffd4f7e2/Q2_d3.png)
![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/576d3266-3bfe-457e-bed3-3f2bb3eafda6/Q2_d3_tempogram.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/576d3266-3bfe-457e-bed3-3f2bb3eafda6/Q2_d3_tempogram.png)

***P-score & ALOTC score of 8 Genres. [2T1, 2T2]***
![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f7d9809d-95f8-4a0b-9f94-2f1191641887/Q2_m2.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f7d9809d-95f8-4a0b-9f94-2f1191641887/Q2_m2.png)
![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d49c0e48-27f8-4141-a4c2-ee7b12a08e68/Q2_m2_tempogram.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d49c0e48-27f8-4141-a4c2-ee7b12a08e68/Q2_m2_tempogram.png)

***P-score & ALOTC score of 8 Genres. [3T1, 3T2]***
![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e86248f3-1779-4304-998a-b4355ad68857/Q2_m3.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e86248f3-1779-4304-998a-b4355ad68857/Q2_m3.png)
![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/61ed3829-2702-43c9-90fa-64c95b3619bb/Q2_m3_tempogram.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/61ed3829-2702-43c9-90fa-64c95b3619bb/Q2_m3_tempogram.png)

## Q3

### Problem Statement

- Using **madmom** to estimate tempo on the Ballroom dataset. Evaluate the performance using again the P-score and the ALOTC score. Note that **madmom** might output more than two tempi with their saliency values; you may select the output tempi with the two highest saliency values as the results. The saliency values should be normalized to 1 before computing the P-score.

## Q4

### Problem Statement

- Using **librosa.beat.beat**_track to find the beat positions of a song. Evaluate this beat tracking algorithm on the Ballroom dataset.

### Results

***F-score of 8 Genres by using librosa.beat.beat.***
![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9735cb19-1441-475b-8cc3-64b38e37388c/Q4-.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9735cb19-1441-475b-8cc3-64b38e37388c/Q4-.png)

## Q5

### Problem Statement

- Also use this algorithm on the SMC dataset and the JCS dataset. Compare the results to the Ballroom dataset. Could you explain the difference in performance?

### Results

***F-score of SMC & JCS by using madmom.features.beats.***
![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/caef2f50-4e76-452c-b1ea-dbbc00303fb0/Q5-.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/caef2f50-4e76-452c-b1ea-dbbc00303fb0/Q5-.png)
![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/36156531-e18f-48cc-8676-8f2f06b76eda/Q5_JCS-.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/36156531-e18f-48cc-8676-8f2f06b76eda/Q5_JCS-.png)

## Q6

### Problem Statement

- Use any function in **madmom.features.beats** for beat tracking and downbeat tracking in the Ballroom and the JCS dataset, and for beat tracking for the SMC dataset (note: there is no downbeat annotation in the SMC dataset). For downbeat tracking, also compute the same F-score using tolerance of ±70 ms. Compare the results to the ones in Q3 and Q4. How much improvement it gains?





