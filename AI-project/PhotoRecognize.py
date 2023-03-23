from deepface import DeepFace
import pandas as pd
import numpy as np

img_path = 'data/face_img/zdj.jpg'  # 输入：图片路径

obj = DeepFace.analyze(img_path=img_path, actions=['emotion'])
print(obj)
emotion_list = [
    obj['emotion']['happy'],
    obj['emotion']['sad'],
    obj['emotion']['neutral'],
    obj['emotion']['fear'],
    obj['emotion']['angry'],
]
emotion_list = (emotion_list-np.min(emotion_list))/(np.max(emotion_list)-np.min(emotion_list))  # 最值归一化
print(emotion_list)

csv_file_all = pd.read_csv('./data/predict_music_output.csv')   # 这个文件是死的，是我的训练结果
data = np.array(csv_file_all.loc[:, 'HAPPY':])

distance_list = []
for line in data:
    print(emotion_list, line)
    dist = np.linalg.norm(emotion_list - line)  # 求相减向量的范数
    distance_list.append(dist)

print('distance_list=', distance_list)

# 取到误差最小的下标
imin = np.argmin(distance_list)

# 找出误差最小的音乐id
csv_file_all = np.array(csv_file_all)
best_id = csv_file_all[imin][0]

print('best_id = '+str(best_id))       # 输出：音频路径


