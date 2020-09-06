from pandas import read_csv
from datetime import datetime
import os
import sys

os.chdir(sys.path[0])

# load data
def parse(x):
	return datetime.strptime(x, '%Y %m %d %H')

# 从csv文件载入数据
#parse_dates:这是指定含有时间数据信息的列
#index_col=0表示将第0列作为索引列
#date_parser：指定将输入的字符串转换为可变的时间数据。Pandas默认的数据读取格式是‘YYYY-MM-DD 
dataset = read_csv('PRSA_data_2010.1.1-2014.12.31.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)

#丢弃第一列
dataset.drop('No', axis=1, inplace=True)
# 手动指定行标题
dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
#指定索引名
dataset.index.name = 'date'
# 将所有“NA”替换为0
dataset['pollution'].fillna(0, inplace=True)
# 丢弃前24小时的数据
dataset = dataset[24:]
# summarize first 5 rows
print(dataset.head(5))
# save to file
dataset.to_csv('pollution.csv')
