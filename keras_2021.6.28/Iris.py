#  -*-coding:utf8 -*-

"""
Created on 2021 6 28

@author: 陈雨
"""

'''基于keras的Tensorflow 2.0分类问题案例'''

import tensorflow as tf
import keras
import numpy as np
from sklearn.datasets import load_iris

# 下载完以后加载数据集
data = load_iris()
iris_target = data.target
iris_data = np.float32(data.data)

iris_target = np.float32(
    tf.keras.utils.to_categorical(iris_target, num_classes=3))

# 处理三类数据集 对其标签进行独热编码
iris_data = tf.data.Dataset.from_tensor_slices(iris_data).batch(50)
iris_target = tf.data.Dataset.from_tensor_slices(iris_target).batch(50)

model = tf.keras.models.Sequential()

# 将一个一个layer加入模型
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(3, activation='softmax'))

#优化器adam算法
opt = tf.optimizers.Adam(0.001)

for epoch in range(1000):
    for _data, lable in zip(iris_data, iris_target):
        with tf.GradientTape() as tape:
            logits = model(_data)
            loss_value = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(
                y_true=lable, y_pred=logits))
                
            # 更新梯度函数
            grads = tape.gradient(loss_value, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
    print('Training loss:', loss_value.numpy())
