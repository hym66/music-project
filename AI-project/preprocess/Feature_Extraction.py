import librosa
import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import csv

import logging
from utils import mp3_path
SAMPLE_INTERVAL = 30
f_ouput = open('../data/raw_Emotion_features.csv', 'w', newline="")
writer = csv.writer(f_ouput)
writer.writerow(['id','song_name','tempo','total_beats','average_beats','chroma_stft_mean','chroma_stft_std','chroma_stft_var','chroma_cq_mean','chroma_cq_std','chroma_cq_var','chroma_cens_mean','chroma_cens_std','chroma_cens_var','melspectrogram_mean','melspectrogram_std','melspectrogram_var','mfcc_mean','mfcc_std','mfcc_var','mfcc_delta_mean','mfcc_delta_std','mfcc_delta_var','rmse_mean','rmse_std','rmse_var','cent_mean','cent_std','cent_var','spec_bw_mean','spec_bw_std','spec_bw_var','contrast_mean','contrast_std','contrast_var','rolloff_mean','rolloff_std','rolloff_var','poly_mean','poly_std','poly_var','tonnetz_mean','tonnetz_std','tonnetz_var','zcr_mean','zcr_std','zcr_var','harm_mean','harm_std','harm_var','perc_mean','perc_std','perc_var','frame_mean','frame_std','frame_var'])
f_ouput.flush()

def extract_feature(path):
    id = 1  # Song ID

    # Traversing over each file in path
    file_data = [f for f in listdir(path) if isfile(join(path, f))]
    for line in file_data:
        if (line[-1:] == '\n'):
            line = line[:-1]

        # Reading Song
        songname = path + line
        time = librosa.get_duration(filename=songname)
        print(time)

        try:
            y, sr = librosa.load(songname)   # y：信号值，sr：采样率
            S = np.abs(librosa.stft(y))


            # Extracting Features
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
            chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
            melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
            rmse = librosa.feature.rms(y=y)
            cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            poly_features = librosa.feature.poly_features(S=S, sr=sr)
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            harmonic = librosa.effects.harmonic(y)
            percussive = librosa.effects.percussive(y)

            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            mfcc_delta = librosa.feature.delta(mfcc)

            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            frames_to_time = librosa.frames_to_time(onset_frames[:20], sr=sr)

            print('写入一行！')
            writer.writerow([id, str(line[:-4]) + ".mp3", tempo, sum(beats), np.average(beats), np.mean(chroma_stft), np.std(chroma_stft), np.var(chroma_stft), np.mean(chroma_cq)
                                , np.std(chroma_cq), np.var(chroma_cq), np.mean(chroma_cens), np.std(chroma_cens), np.var(chroma_cens), np.mean(melspectrogram), np.std(melspectrogram), np.var(melspectrogram), np.mean(mfcc), np.std(mfcc)
                                , np.var(mfcc), np.mean(mfcc_delta), np.std(mfcc_delta), np.var(mfcc_delta), np.mean(rmse), np.std(rmse), np.var(rmse)
                                , np.mean(cent), np.std(cent), np.var(cent), np.mean(spec_bw), np.std(spec_bw), np.var(spec_bw), np.mean(contrast)
                                , np.std(contrast), np.var(contrast), np.mean(rolloff), np.std(rolloff), np.var(rolloff), np.mean(poly_features), np.std(poly_features)
                                , np.var(poly_features), np.mean(tonnetz), np.std(tonnetz), np.var(tonnetz), np.mean(zcr), np.std(zcr), np.var(zcr)
                                , np.mean(harmonic), np.std(harmonic), np.var(harmonic), np.mean(percussive), np.std(percussive), np.var(percussive)
                                , np.mean(frames_to_time), np.std(frames_to_time), np.var(frames_to_time)])
            f_ouput.flush()
        except Exception as e:
            print(e, songname)
            logger.warning("出错！"+str(e)+str(songname))
            continue

        print(songname[:-4] + "-" + ".mp3")
        id = id + 1


logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("log-FeatureExtraction.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info("Start print log")
print('开始特征提取！')
extract_feature(mp3_path)
print('特征提取全部完成！')
logger.info("Finish")

f_ouput.close()