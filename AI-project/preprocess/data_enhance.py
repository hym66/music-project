# coding=gbk

import logging
import os
import librosa
import numpy as np
import random
import matplotlib.pyplot as plt
import librosa.display
from scipy.io import wavfile

from utils import mp3_path

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("log-DataEnhance.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info("Start print log")

def Shift_Wave(y, sr):
    n_steps = random.randint(-10, 10)

    y_ps = librosa.effects.pitch_shift(y, sr, n_steps)  # ���Ĳ�����n_steps

    plt.subplot(511)

    # librosa.display.waveplot(y_ps, sr=sr)

    # plt.title('Pitch Shift transformed waveform')

    return y_ps


def Stretch_Wave(y):
    rate = random.uniform(0.8, 1.2)

    y_ts = librosa.effects.time_stretch(y, rate)  # ���Ĳ�����rate

    # plt.subplot(513)

    # librosa.display.waveplot(y_ts, sr=sr)

    plt.title('Time Stretch transformed waveform')

    return y_ts




path = mp3_path
files = os.listdir(path)
files = [path + f for f in files if f.endswith('.mp3')]

def getFileName(path):
    content = path.split('/')
    content = content[len(content)-1]
    content = content.split('.')
    content = content[0]
    return content

for i in range(len(files)):
    try:
        # ������Ҫ����������������Ƶ

        FileName = files[i]

        print("Shift Wave File Name is ", FileName)
        time = librosa.get_duration(filename=FileName)
        # print(time-1)
        y, sr = librosa.load(FileName, duration=time)
        # print(sr)

        plt.subplot(515)

        # librosa.display.waveplot(y, sr=sr)

        # ����Shift�������wav�ļ��������Ƽ�����·��

        save_name = getFileName(FileName) + '-shift' + '.mp3'

        # print(save_name)

        save_path_shift = mp3_path

        path_shift = save_path_shift + save_name

        # print(path_shift)

        # ����shift�ź�����

        data_shift = Shift_Wave(y, sr)

        # ����stretch�������wav�ļ��������Ƽ�����·��

        save_name = getFileName(FileName) + '-stretch' + '.mp3'

        # print(save_name)

        save_path_stretch = mp3_path

        path_stretch = save_path_stretch + save_name

        # print(path_stretch)

        # ����stretch�ź�����

        data_stretch = Stretch_Wave(y)

        # Saving the audio

        librosa.output.write_wav(path_shift, data_shift, sr)

        librosa.output.write_wav(path_stretch, data_stretch, sr)
    except Exception as e:
        print(e)
        logger.warning("����" + str(e))
        continue

logger.info("Finish")
print('run over��')