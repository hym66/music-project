import pandas as pd
import csv

input_file = open('../data/raw_Emotion_features.csv', 'r', errors='ignore')
output_file = open('../data/Emotion_features_split.csv', 'w', newline="", errors='ignore')

input_reader = csv.reader(input_file)
output_writer = csv.writer(output_file)

print('开始筛除数据......')
i = 0
for line in input_reader:
    if i == 0:
        output_writer.writerow(line)
        i += 1
        continue

    skip = False
    for item in line:
        if item == 0 or item == '' or item == None or item == 'nan' or item == '0':
            skip = True
            break

    if not skip:
        output_writer.writerow(line)

input_file.close()
output_file.close()
print('筛除完成！')