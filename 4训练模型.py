# -*- coding: UTF-8 -*-
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
import pandas as pd
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt

h5=pd.HDFStore("Preprocessing","r")
reframed=h5["data"]

# 将数据分割为训练集和测试集
values = reframed.values
n_train_hours = round(0.7 * values.shape[0])
train = values[:n_train_hours, :]
print(train.shape)
test = values[n_train_hours:, :]
print(test.shape)

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)



# design network
model = Sequential()
#inputshape=(,): 当把这层作为某个模型的第一层时，需要用到该参数，输入矩阵形状：(n_samples, dim_input)，dim_input指影响输出的特征数，单因子模型所以填1，当建立多因子模型时填因子个数
#output_dim=50: 输出长度为50的向量，因为每个时间步都输出一次，共50个时间步，所以最后得到50*50的序列，至于输出整个序列还是最后一个序列取决于参数return_sequences
#return_sequences：默认False，控制返回类型。若为True则返回整个序列，否则仅返回输出序列的最后一个输出
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
print('lstm层输出的矩阵',model.output_shape)

#此处输出数据形状应该与y_train的形状一致，否则会报错
model.add(Dense(1))
print('全连接层输出',model.output_shape)


#编译模型时, 我们需要声明损失函数和优化器 (SGD, Adam 等等)
#optimizer：优化器，该参数可指定为已预定义的优化器名，如rmsprop、adagrad
#loss：损失函数,该参数为模型试图最小化的目标函数，它可为预定义的损失函数名，如categorical_crossentropy、mse
#metrics：指标列表,对分类问题，我们一般将该列表设置为metrics=['accuracy']
model.compile(loss='mae', optimizer='adam')

#训练模型
#batch_size：整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。
#nb_epochs：整数，训练的轮数，训练数据将会被遍历nb_epoch次。Keras中nb开头的变量均为”number of”的意思
#verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
#validation_split : 验证数据的使用比例。
#validation_data : 被用来作为验证数据的(X, y)元组。会代替validation_split所划分的验证数据。
#shuffle : 是否对每一次迭代的样本进行shuffle操作。’batch’是一个用于处理HDF5（keras用于存储权值的数据格式）数据的特殊选项。
#show_accuracy:每次迭代是否显示分类准确度。
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# 保存模型
model.save('model.h5')   # HDF5文件，pip install h5py

# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
