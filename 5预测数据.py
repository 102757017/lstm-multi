# -*- coding: UTF-8 -*-
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
import pandas as pd
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from numpy import concatenate
import math
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import sys

os.chdir(sys.path[0])

f=np.load('Preprocessing.npz')
train_X=f['train_X']
train_y=f['train_y']
test_X=f['test_X']  
test_y=f['test_y']



# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)




model = load_model('model.h5')   # HDF5文件，pip install h5py




#预测数据
yhat = model.predict(test_X)

dbfile = open('pickle_scaler', 'rb')
scaler = pickle.load(dbfile)
dbfile.close()

inv_yhat = scaler.inverse_transform(yhat)
inv_y = scaler.inverse_transform(test_y)


# 计算均方根误差 RMSE
rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

plt.plot(inv_y,label="true")
plt.plot(inv_yhat,label="predict")

leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)
plt.show()

