CLASS_NUM = 5  # 分类数
random_seed = 7
eachInput_rows = 1
eachInput_cols = 40 # 总通道数是51
SPLIT_RATIO = 0.8

mp3_path = 'G:/music/music_all/'

w_to_id = {    # 字典映射
    'happy':0,
    'sad':1,
    'tender':2,
    'fear':3,
    'angry':4,
}