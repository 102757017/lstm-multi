# -*- coding: UTF-8 -*-
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np
import os
import sys

os.chdir(sys.path[0])

#将数据集转换为监督学习问题
#data：输入numpy矩阵,行为日期，列为特征值
#n_in：基于多少个周期的数据来预测后面的数据。值可能介于[1..len（data）]
#n_out：预测后多少个周期的数据。值可以在[0..len（data）-1]之间。
#dropnan：是否删除具有NaN值的行。
def series_to_supervised(data, n_in=50, n_out=1, dropnan=True):
    #单因子数据输入为list，多因子数据输入为矩阵
    if type(data) is list:
        n_vars=1
    else:
        n_vars=data.shape[1]
    #n_vars为特征个数
    print("特征个数n_vars为:",n_vars)
        
    #将矩阵转换为dataframe
    df = DataFrame(data)
    #创建两个空的list
    cols, names = list(), list()
    # 输入序列 (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        #shift函数将dataframe整体向下平移i个单元格创建一个新的dataframe,原dataframe自身不会产生变化，由此缺少的值用NaN填补
        #只有数据内容产生了移动，索引名不会同时产生移动
        cols.append(df.shift(i))

        #创建输入数据的表头
        for j in range(n_vars):
                names.append('var%d(t-%d)' % (j+1, i))

    
    #删除我们不想预测的列
    df.drop(df.columns[[1,2,3,4,5,6,7]], axis=1, inplace=True)
    
    # 预测序列 (t, t+1, ... t+n)
    for i in range(0, n_out):
        #将dataframe整体向上平移i个单元格，由此缺少的值用NaN填补
        cols.append(df.shift(-i))
        #创建预测数据的表头,range内的数为train_y的列数
        for j in range(1):
            names.append('var%d(t+%d)' % (j+1, i))
            
    # cols为多个dataframe组成了列表，concat函数将这些dataframe按axis=1的方向连接起来，组成了一个新的dataframe
    agg = concat(cols, axis=1)
    
    #names为1个list，将其作为agg的标题
    agg.columns = names
    
    # 丢弃agg中有NaN的行
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# 从csv文件载入数据，header=0表示将第0行作为标题栏，index_col=0表示将第0列作为索引列
dataset = read_csv('pollution.csv', header=0, index_col=0)


#将DataFrame转换为numpy矩阵，并且舍弃标题栏和索引列
values = dataset.values

# 第5列数据为风向，将string类型的标签值统一转换成range(标签值个数-1)范围内数值
encoder = LabelEncoder()
#fit_transform本质是先调用fit然后调用transform
values[:,4] = encoder.fit_transform(values[:,4])


# 将numpy矩阵元素转换为浮点数
values = values.astype('float32')


#将数据集转换为监督学习问题,返回值为dataframe
reframed = series_to_supervised(values, 7, 1)


print(reframed.shape)
print(reframed)

#reframed.to_csv("Preprocessing.csv")

# 将数据分割为训练集和测试集
values = reframed.values
n_train_hours = round(0.7 * values.shape[0])
n_validation_hours = round(0.9 * values.shape[0])
train = values[:n_train_hours, :]
validation=values[n_train_hours:n_validation_hours, :]
print(train.shape)
test = values[n_validation_hours:, :]
print(test.shape)


#np.random.shuffle(train)
# 将训练集和测试集分割为输入数据x和输出数据y
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
validation_X, validation_y = validation[:, :-1], validation[:, -1]

train_y=train_y.reshape(-1, 1)
test_y=test_y.reshape(-1, 1)
validation_y=validation_y.reshape(-1, 1)

print("train_X",train_X.shape)
print("train_y",train_y.shape)
print("test_X",test_X.shape)
print("test_y",test_y.shape)
print("validation_X",validation_X.shape)
print("validation_y",validation_y.shape)


# 归一化,将数据缩放至给定的最小值与最大值之间，通常是０与１之间，用MinMaxScaler实现
scaler = MinMaxScaler(feature_range=(0, 1)).fit(train_X)
train_X = scaler.transform(train_X)

scaler = MinMaxScaler(feature_range=(0, 1)).fit(train_y)
train_y = scaler.transform(train_y)

#将scaler对象保存为文件
dbfile = open('pickle_scaler', 'wb')
pickle.dump(scaler, dbfile)
dbfile.close()

scaler = MinMaxScaler(feature_range=(0, 1)).fit(test_X)
test_X = scaler.transform(test_X)

scaler = MinMaxScaler(feature_range=(0, 1)).fit(test_y)
test_y = scaler.transform(test_y)

scaler = MinMaxScaler(feature_range=(0, 1)).fit(validation_X)
validation_X = scaler.transform(validation_X)

scaler = MinMaxScaler(feature_range=(0, 1)).fit(validation_y)
validation_y = scaler.transform(validation_y)

#将数据以压缩格式存储
np.savez('Preprocessing',train_X=train_X,train_y=train_y,validation_X=validation_X,validation_y=validation_y,test_X=test_X,test_y=test_y)
