from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

#将数据集转换为监督学习问题
#data：输入numpy矩阵
#n_in：作为输入（X）的滞后观测值的数量。值可能介于[1..len（data）]可选。默认为1
#n_out：作为输出的观察次数（y）。值可以在[0..len（data）-1]之间。可选的。默认为1
#dropnan：是否删除具有NaN值的行。
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    #单因子数据输入为list，多因子数据输入为矩阵
	if type(data) is list:
	    n_vars=1
	else:
	    n_vars=data.shape[1]
	#矩阵转换为dataframe
	df = DataFrame(data)
	#创建两个空的list
	cols, names = list(), list()
	# 输入序列 (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	for j in range(n_vars):
		names.append('var%d(t-%d)' % (j+1, i))

	# 预测序列 (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			for j in range(n_vars):
				names.append('var%d(t-%d)' % (j+1, i))
		else:
			for j in range(n_vars):
				names.append('var%d(t+%d)' % (j+1, i))
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
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



# 归一化,将数据缩放至给定的最小值与最大值之间，通常是０与１之间，用MinMaxScaler实现
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
print(scaled)

#将数据集转换为监督学习问题,返回值为dataframe
reframed = series_to_supervised(scaled, 1, 1)

#删除我们不想预测的列
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
print(type(reframed.head()))
print(reframed.head())
