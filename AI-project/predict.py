import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from DRSN_model import DRSN
import tensorflow as tf
import csv
import pandas as pd
from tensorflow.python.keras import backend as K
from utils import *
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from tensorflow.keras.utils import to_categorical

def get_feature_label(path):
    data = pd.read_csv(path)
    feature = data.loc[:, 'tempo':]
    featureName = list(feature)

    # 这里已经做了归一化了
    for name in featureName:
        feature[name] = (feature[name] - feature[name].min()) / (feature[name].max() - feature[name].min())

    features = feature.values
    labels = data.loc[:, 'class'].dropna()



    return features, labels

def getFeatures():
    feature = data.loc[:, 'tempo':]  # data是main函数中的全局变量
    featureName = list(feature)

    # 这里已经做了归一化了
    for name in featureName:
        feature[name] = (feature[name] - feature[name].min()) / (feature[name].max() - feature[name].min())

    # print(feature)
    features = feature.values

    # 处理数据维度
    if K.image_data_format() == 'channels_first':
        features = features.reshape(1, eachInput_rows, features.shape[0], eachInput_cols)
    else:
        features = features.reshape(features.shape[0], eachInput_rows, eachInput_cols, 1)

    return features

def get_feature_important(features, labels, k):
    model = SelectKBest(chi2, k=k)
    features = model.fit_transform(features, labels)

    # 处理数据维度
    if K.image_data_format() == 'channels_first':
        features = features.reshape(1, eachInput_rows, features.shape[0], eachInput_cols)
    else:
        features = features.reshape(features.shape[0], eachInput_rows, eachInput_cols, 1)

    return features

if __name__ == '__main__':
    data = pd.read_csv('./data/predict_music_input.csv')
    # 打开输出文件
    output_file = open('./data/predict_music_output.csv', 'w', newline="", errors='ignore')
    output_writer = csv.writer(output_file)

    # 实例化模型
    model = DRSN()

    # 读取训练的参数
    checkpoint_save_path = './checkpoint/DRSN-new.ckpt'
    if os.path.exists(checkpoint_save_path + '.index'):
        print('------------------load model-------------------')
        model.load_weights(checkpoint_save_path)

    # 获取待预测数据
    features, labels = get_feature_label('data/Emotion_features_ready.csv')
    features = get_feature_important(features, labels, k=eachInput_cols)
    # features = getFeatures()

    # 利用模型进行预测
    result = model.predict(features)

    # 写入结果到输出的csv文件中
    output_writer.writerow(['song_name', 'HAPPY', 'SAD', 'TENDER', 'FEAR', 'ANGER'])

    for i in range(len(result)):
        line = result[i]
        line = np.array(line, dtype=object)
        line = np.insert(line, 0, str(data['song_name'][i]))    # ???
        output_writer.writerow(line)

    # 取最大值作为结果
    pred = tf.argmax(result, axis=1)
    print(result)
    print(pred)


    output_file.close()