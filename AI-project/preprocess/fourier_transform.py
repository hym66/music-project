import os
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import math
plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams.update({"font.size":9})#此处必须添加此句代码方可改变标题字体大小
## import data
df = pd.read_csv('E:\\大二下\\专业实习\\我的文件\\assets\\data_all.csv', parse_dates=['time'],encoding='gbk')
print(df.head())
print(df.tail())
print(df.shape)
print(df.columns)
temp=df.groupby('time',as_index=False)['time','chl','vo','uo','no3','po4','si','nppv'].mean().round(2)
# Create chl concentration plot
#temp=df.groupby(['time','lat','lon'],as_index=False).mean().round(2)
df=temp
## https://www.earthdatascience.org/courses/use-data-open-source-python/use-time-series-data-in-python/date-time-types-in-pandas-python/customize-dates-matplotlib-plots-python/
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(df['time'],df['chl'], label='chl concentration')
ax.set(xlabel="Date",
       ylabel="mg/m-3",
       title="chlorophyll concentration ")
date_form = DateFormatter("%Y")
ax.xaxis.set_major_formatter(date_form)
plt.show()


#Drop the first 21 rows
#For doing the fourier
#dataset = T_df.iloc[20:,:].reset_index(drop=True)
dataset=df
#Getting the Fourier transform features
def FFT(dataset):
    data_FT = dataset[['time', 'chl']]
    dt=0.01
    t=dataset['time']
    data_FT=data_FT.reset_index(drop=False)
    n=904
    # w=2*np.pi*np.array([3,6,9])
    # f=np.sum(np.sin(w[:,None]*t1),0)
    # fn=f+3*np.random.randn(n)
    h=np.fft.fft(data_FT['chl'])
    PSD=h*np.conj(h)/n
    freq=(1/(dt*n))*np.arange(n)
    PSD0=np.where(PSD<1,0,PSD)
    h = np.where(PSD < 1, 0, h)
    H=np.fft.ifft(h)
    fig,ax=plt.subplots(2,1)

    m=n//2
    ax[0].plot(freq[:m],PSD[:m],label='Noise')
    ax[0].plot(freq[:m], PSD0[:m],c='k',label='Clean')
    ax[0].axhline(1,ls='--',c='r')
    ax[1].plot(t, data_FT['chl'], label='real')
    ax[1].plot(t,H,c='k',label='Denoise')

    plt.show()
H=FFT(dataset)
def get_fourier_transfer(dataset):
    # Get the columns for doing fourier
    data_FT = dataset[['time', 'chl']]

    close_fft = np.fft.fft(np.asarray(data_FT['chl'].tolist()))
    fft_df = pd.DataFrame({'fft': close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

    fft_list = np.asarray(fft_df['fft'].tolist())
    fft_com_df = pd.DataFrame()
    for num_ in [3, 6, 9]:
        fft_list_m10 = np.copy(fft_list);
        fft_list_m10[num_:-num_] = 0
        fft_ = np.fft.ifft(fft_list_m10)
        fft_com = pd.DataFrame({'fft': fft_})
        fft_com['absolute of ' + str(num_) + ' comp'] = fft_com['fft'].apply(lambda x: np.abs(x))
        fft_com['angle of ' + str(num_) + ' comp'] = fft_com['fft'].apply(lambda x: np.angle(x))
        fft_com = fft_com.drop(columns='fft')
        fft_com_df = pd.concat([fft_com_df, fft_com], axis=1)

    return fft_com_df

#Get Fourier features
dataset_F = get_fourier_transfer(dataset)
Final_data = pd.concat([dataset, dataset_F], axis=1)


print(Final_data.head())


Final_data.to_csv("Finaldata_with_Fourier.csv", index=False)
print('成功转化为最终的csv文件!!!!!!!!!!!!')


def plot_Fourier(dataset):
    data_FT = dataset[['time', 'chl']]

    close_fft = np.fft.fft(np.asarray(data_FT['chl'].tolist()))
    fft_df = pd.DataFrame({'fft': close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

    fft_list = np.asarray(fft_df['fft'].tolist())
    plt.figure(figsize=(14, 7), dpi=100)
    fft_list = np.asarray(fft_df['fft'].tolist())
    for num_ in [3, 6, 9]:
        fft_list_m10 = np.copy(fft_list);
        fft_list_m10[num_:-num_] = 0
        plt.plot(np.fft.ifft(fft_list_m10), label='Fourier transform with {} components'.format(num_))
    #plt.plot(data_FT['chl/mg m-3'], label='Real')
    plt.xlabel('Days')
    plt.ylabel('mg/m-3')
    plt.title('chl concentration & Fourier transforms')
    plt.legend()
    plt.show()

plot_Fourier(dataset)
