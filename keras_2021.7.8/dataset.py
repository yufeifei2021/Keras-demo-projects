'''Keras的高级特性:

允许简单而快速的原型设计(由于用户友好，高度模块化，可扩展性)。
同时支持卷积神经网络和循环神经网络，以及两者的组合。
在 CPU 和 GPU 上无缝运行。'''


# Keras内置数据集
# Keras内置波斯顿房价回归数据集、IMDB电影影评情感分类数据集、路透社新闻专线主题分类数据集、手写数字MNIST数据集、时尚MNIST数据库(鞋服裙帽)、CIFAR10小图像数据集和CIFAR100小图像数据集。

# 1 波斯顿房价回归数据集
# 数据集取自卡内基梅隆大学维护的StatLib库。

from tensorflow.python import keras
(x_train, y_train), (x_test, y_test) = keras.datasets.boston_housing.load_data()

# 2 IMDB电影影评情感分类
# 训练集：25000条评论，正面评价标为1，负面评价标为0
# 测试集：25000条评论 
# path="imdb.npz",

from tensorflow.python import keras
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data()

# 3 路透社新闻专线主题分类
# 总数据集：11228条新闻专线，46个主题。
# path="reuters.npz", ....

from tensorflow.python import keras
(x_train, y_train), (x_test, y_test) = keras.datasets.reuters.load_data()

# 4 手写数字MNIST数据集
# 训练集：60000张灰色图像，大小28*28，共10类（0-9）
# 测试集：10000张灰色图像，大小28*28

from tensorflow.python import keras
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 5 时尚MNIST数据库(鞋服裙帽)
# MNIST已经被玩坏了！用时尚MNIST替换吧!
# 训练集：60000张灰色图像，大小28*28，共10类（0-9）
# 测试集：10000张灰色图像，大小28*28

from tensorflow.python import keras
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 6 CIFAR10小图像
# 训练集：50000张彩色图像，大小32*32，被分成10类
# 测试集：10000张彩色图像，大小32*32

from tensorflow.python import keras
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# 7 CIFAR100小图像
# 训练集：50000张彩色图像，大小32*32，被分成100类
# 测试集：10000张彩色图像，大小32*32

from tensorflow.python import keras
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()