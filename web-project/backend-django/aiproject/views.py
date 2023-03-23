from django.http import HttpResponse
from deepface import DeepFace
import pandas as pd
import numpy as np
import os
from pathlib import Path


def hello(request):
    image = request.FILES.get('raw')
    # return HttpResponse(request.FILES)
    print('图片信息为：', image)
    with open("./images/" + image.name, 'wb') as f:
        for c in image.chunks():
            f.write(c)

    # 取后缀名
    # tmpList = image.name.split('.')
    # suffix = tmpList[len(tmpList)-1]


    img_path = "./images/photo." + 'jpg'
    # 重命名
    my_file = Path(img_path)
    if my_file.is_file():
        os.remove(img_path)
    os.rename("./images/" + image.name, img_path)  # 重命名,覆盖原先的名字

    # tmp_file = os.path.join(settings.MEDIA_ROOT, path)


    # img_path = request.GET.get('path')  # 输入：图片路径
    print(img_path)
    obj = DeepFace.analyze(img_path=img_path, actions=['emotion'])
    print(obj)
    emotion_list = [
        obj['emotion']['happy'],
        obj['emotion']['sad'],
        obj['emotion']['neutral'],
        obj['emotion']['fear'],
        obj['emotion']['angry'],
    ]
    emotion_list = (emotion_list - np.min(emotion_list)) / (np.max(emotion_list) - np.min(emotion_list))  # 最值归一化
    print(emotion_list)

    csv_file_all = pd.read_csv('./data/predict_music_output.csv')  # 这个文件是死的，是我的训练结果
    data = np.array(csv_file_all.loc[:, 'HAPPY':])

    distance_list = []
    for line in data:
        dist = np.linalg.norm(emotion_list - line)  # 求相减向量的范数
        distance_list.append(dist)

    # 取到误差最小的下标
    imin = np.argmin(distance_list)

    # 找出误差最小的音乐song_name
    csv_file_all = np.array(csv_file_all)

    best_song_name = csv_file_all[imin][0]
    nameList = best_song_name.split('-')
    best_song_name = nameList[0] + '-' + nameList[1]

    music_path = 'G:/music/music_all/' + best_song_name + '.mp3'

    print('best_song_name = ' + str(best_song_name))  # 输出：音频路径

    os.system(music_path)
    return HttpResponse('best_song_name = ' + best_song_name)
