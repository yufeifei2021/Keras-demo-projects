#  -*-coding:utf8 -*-

"""
Created on 2021 7 2

@author: 陈雨
"""

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
tf.compat.v1.disable_eager_execution()
from keras.utils import np_utils

'''
字母表的一个简单的序列预测问题
'''

np.random.seed(7)

# 原生数据
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# 转为数字
data_len = 26
time_steps = 3
X = list(range(data_len))
y = X[time_steps:]

# 将数据转化为[样本数, 时间步数, 特征数]的形式
# X.shape: [samples, time_step, features]
# y.shape: [samples, one_hot_encodes]
XX = [X[i:i+time_steps] for i in range(data_len-time_steps)] # [samples, time steps * features]
XXX = np.reshape(XX, (data_len - time_steps, time_steps, -1)) # [samples, time steps, features]


# 归一化
# 数值范围变为0～1，这是LSTM网络使用的s形激活函数（sigmoid）的范围。
X = XXX / data_len
# one-hot编码
#y = np_utils.to_categorical(dataY)
y = np.eye(data_len)[y]

model = Sequential()
model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax')) # 输出各类的概率(softmax)
model.compile(loss='categorical_crossentropy',     # 单标签，多分类(categorical_crossentropy)
              optimizer='adam', 
              metrics=['accuracy'])

model.fit(X, y, epochs=500, batch_size=1, verbose=2)

#检查模型在测试集上的表现是否良好
test_loss, test_acc = model.evaluate(X, y)
print('test_acc:', test_acc)

