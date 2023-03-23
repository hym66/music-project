import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
import os
from DRSN_model import DRSN
from utils import *
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

path = 'data/Emotion_features_ready.csv'

def show_train_img(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def get_train_test(features, labels):
    # 标签通过字典映射为数字
    tmp = []
    for i in labels:
        tmp.append(w_to_id[i])
    labels = tmp

    # 打乱顺序
    np.random.seed(random_seed)
    np.random.shuffle(features)
    np.random.seed(random_seed)
    np.random.shuffle(labels)
    tf.random.set_seed(random_seed)

    # 划分训练集与测试集
    len_train = int(len(features) * SPLIT_RATIO)
    len_test = len(features) - len_train

    # 训练集
    train_d = features[:len_train]
    train_l = labels[:len_train]

    train_d = np.array(train_d)
    train_l = np.array(train_l)


    # 测试集
    test_d = features[-len_test:]
    test_l = labels[-len_test:]

    test_d = np.array(test_d)
    test_l = np.array(test_l)


    # 使用one-hot编码将结果向量化(对标签数据向量化我采用的是from tensorflow.keras.utils import to_categorical）直接可转化
    train_l = to_categorical(train_l)
    test_l = to_categorical(test_l)

    return (train_d, train_l), (test_d, test_l)

def get_feature_label(path):
    data = pd.read_csv(path)
    feature = data.loc[:, 'tempo':]
    featureName = list(feature)

    # 这里已经做了归一化了
    for name in featureName:
        feature[name] = (feature[name] - feature[name].min()) / (feature[name].max() - feature[name].min())

    plt.style.use('ggplot')

    features = feature.values
    labels = data.loc[:, 'class'].dropna()

    return features, labels

def get_feature_important(features, labels, k):
    model = SelectKBest(chi2, k=k)
    X_new = model.fit_transform(features, labels)
    return X_new


if __name__ == '__main__':
    # 获取数据集
    features, labels = get_feature_label(path)
    features = get_feature_important(features, labels, k=eachInput_cols)
    (x_train, y_train), (x_test, y_test) = get_train_test(features, labels)

    # 处理数据维度
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(1, eachInput_rows, x_train.shape[0], eachInput_cols)
        x_test = x_test.reshape(1, eachInput_rows, x_test.shape[0], eachInput_cols)
        input_shape = (1, eachInput_rows, eachInput_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], eachInput_rows, eachInput_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], eachInput_rows, eachInput_cols, 1)
        input_shape = (eachInput_rows, eachInput_cols, 1)

    # 模型实例化
    model = DRSN()

    # 配置
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    # 读取之前训练的模型
    checkpoint_save_path = './checkpoint/DRSN-new.ckpt'
    if os.path.exists(checkpoint_save_path + '.index'):
        print('------------------load model-------------------')
        model.load_weights(checkpoint_save_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=True)

    # 训练
    history = model.fit(x_train, y_train, batch_size=16, epochs=120, verbose=1, validation_data=(x_test, y_test),
                        shuffle=True, callbacks=[cp_callback])

    # 打印网络结构
    model.summary()

    # 显示训练集和验证集的acc和loss曲线
    show_train_img(history)