import pandas as pd
import csv

input_file = open('../data/Emotion_features.csv', 'r', errors='ignore')
output_file = open('../data/Emotion_features_ready.csv', 'w', newline="", errors='ignore')

input_reader = csv.reader(input_file)
output_writer = csv.writer(output_file)

print('开始筛除数据......')
max = 800
i1, i2, i3, i4, i5 = 0, 0, 0, 0, 0
for line in input_reader:
    if line[2] == 'angry':
        i1 += 1
        if i1 < max:
            output_writer.writerow(line)
        continue
    if line[2] == 'fear':
        i2 += 1
        if i2 < max:
            output_writer.writerow(line)
        continue
    if line[2] == 'happy':
        i3 += 1
        if i3 < max:
            output_writer.writerow(line)
        continue
    if line[2] == 'sad':
        i4 += 1
        if i4 < max:
            output_writer.writerow(line)
        continue
    if line[2] == 'tender':
        i5 += 1
        if i5 < max:
            output_writer.writerow(line)
        continue
    else:
        output_writer.writerow(line)

input_file.close()
output_file.close()
print('筛除完成！')