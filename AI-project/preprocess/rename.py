import os
mood_list = ['happy', 'sad', 'tender', 'angry', 'fear']


for mood in mood_list:
    f_path = "G:/music/music/" + mood
    flist = os.listdir(f_path)

    print('开始重命名'+mood+'文件夹，请耐心等待......')
    i = 0

    for f in flist:
        i += 1
        src = os.path.join(os.path.abspath(f_path), f) #原先的图片名字
        dst = os.path.join(os.path.abspath(f_path), mood+ '-' +str(i) + '.mp3') #根据自己的需要重新命名,可以把str(i)+'.jpg'改成你想要的名字
        os.rename(src, dst) #重命名,覆盖原先的名字

    print('重命名'+mood+'文件夹完成！')

print('所有文件重命名完成！')