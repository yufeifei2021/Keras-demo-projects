#  -*-coding:utf8 -*-

"""
Created on 2021 7 8

@author: 陈雨
"""


'''MNIST数据集是一个手写数字图片的数据集，其包含了60000张训练图片
和10000张测试图片，这些图片是28×28的灰度图片，共包含0到9总计10个数字
'''
from keras import layers
from keras import models
import keras
from keras.datasets import mnist
# 加载数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 数据集的相关信息
print('shape of train images is ', train_images.shape)
print('shape of train labels is ', train_labels.shape)
print('train labels is ', train_labels)
print('shape of test images is ', test_images.shape)
print('shape of test labels is', test_labels.shape)
print('test labels is', test_labels)

'''设计网络结构(在Keras中layers是网络结构的基石)'''

network = models.Sequential()

# layers.Dense()的第一个参数指定的是当前层的输出维度
# 全连接层
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))

# softmax层，返回10个概率，每个概率表示表示当前图片属于数字的概率
network.add(layers.Dense(10, activation='softmax'))

# 指定optimizer、loss function和metrics并编译网络
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy', metrics=['accuracy'])

# 处理训练集
train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype('float32')/255

# 处理测试集
test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32')/255

# 处理标签
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 训练模型
print("-------------------训练模型----------------------")
network.fit(train_images,train_labels,epochs=5,batch_size=128)

# 评估测试集
print("-------------------评估数据集----------------------")
test_loss,test_acc = network.evaluate(test_images,test_labels)
print("test_loss:",test_loss)
print("test_acc:",test_acc)


