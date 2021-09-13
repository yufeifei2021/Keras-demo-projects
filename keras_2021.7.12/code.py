#  -*-coding:utf8 -*-

"""
Created on 2021 7 12

@author: 陈雨
"""

'''机器学习进行情感分析'''


#将必要的库导入
import jieba
import pandas as pd
import csv
import time


'''数据清洗'''
# 创建停用词列表
def stopwordslist():
    stopwords = [line.strip() for line in open('/home/chenyu/case_study/keras_2021.7.12/HGD_StopWords.txt',encoding='UTF-8').readlines()]
    return stopwords

# 对句子进行中文分词
def seg_depart(sentence):
    # 对文档中的每一行进行中文分词
    # print("正在分词")
    sentence_depart = jieba.cut(sentence.strip())
    # 引进停用词列表
    stopwords = stopwordslist()
    # 输出结果为outstr
    outstr = ''
    # 去停用词
    for word in sentence_depart:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr


# 给出文档路径
filename = "/home/chenyu/case_study/keras_2021.7.12/ALL_Comment.txt"
outfilename = "/home/chenyu/case_study/keras_2021.7.12/stop_seg_word.txt"
inputs = open(filename, 'r', encoding='UTF-8')
outputs=open(outfilename, 'w', encoding='UTF-8')

# 将输出结果写入out中
start = time.time()
count=0
for line in inputs:
    line_seg = seg_depart(line)
    #writer.writerows(line_seg + '\n') 
    outputs.writelines(line_seg + '\n')
    #print("-------------------正在分词和去停用词-----------")
    count=count+1
print("一共处理了",count,"条数据")
outputs.close()
inputs.close()
end=time.time()
running_time = end - start
print("删除停用词和分词成功！！！")
print("使用时间",running_time)


'''转换数据格式'''
with open("/home/chenyu/case_study/keras_2021.7.12/stop_seg_word.txt") as f:
    lines=f.readlines()
# for line in lines:
#     print(line)


#创建方法对象
data = pd.DataFrame()
data.head

# 将评论数据按行写入data中的“评论”一列
#将txt文件中的数据按行写入csv文件
with open('/home/chenyu/case_study/keras_2021.7.12/stop_seg_word.txt', encoding='utf-8') as f:
    line = f.readlines()
    line = [i.strip() for i in line]
    # print(len(line))
#建立评论这一列，将数据进行循环写入
data['评论'] = line

# 读取评分数据
with open('/home/chenyu/case_study/keras_2021.7.12/All_label.txt', "r",encoding='utf-8') as f:
    all_label=f.readlines()
    # print(all_label)
    print(type(all_label))
    print(len(all_label))
    
# 将评分数据以逗号形式分割   
all_labels=[]
for element in all_label:
    all_labels.extend(element.split(','))


#建立“评分”这一列，将数据进行循环写入
data['评分'] = all_labels

#查看数据
data


#将整理好的数据进行保存，路径根据自己设备指定即可
data.to_csv('/home/chenyu/case_study/keras_2021.7.12/reviews_score_update.csv')


'''机器学习部分'''
#首先将用到的包进行导入
import pandas as pd
import numpy as np
import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection 
from sklearn import preprocessing

#将数据进行读取
data=pd.read_csv('/home/chenyu/case_study/keras_2021.7.12/reviews_score_update.csv',index_col=0)
data.head()

#现在是划分数据集
#random_state 取值
# 这是为了在不同环境中，保证随机数取值一致，以便验证模型的实际效果
train_x,test_x,train_y,test_y=model_selection.train_test_split(data.评论.values.astype('U'),data.评分.values,test_size=0.1,random_state=1)
 
#划分完毕，查看数据形状
print(train_x.shape,test_x.shape)
#train_x 训练集数据 test_x 测试集数据 
#train_y训练集的标签 test_y 测试集的标签

#定义函数，从哈工大中文停用词表里面
# 把停用词作为列表格式保存并返回 
# 在这里加上停用词表是因为TfidfVectorizer和CountVectorizer的函数中
#可以根据提供用词里列表进行去停用词
def get_stopwords(stop_word_file):
    with open(stop_word_file) as f:
        stopwords=f.read()
    stopwords_list=stopwords.split('\n')
    custom_stopwords_list=[i for i in stopwords_list]
    return custom_stopwords_list

#获得由停用词组成的列表
stop_words_file = '/home/chenyu/case_study/keras_2021.7.12/HGD_StopWords.txt'
stopwords = get_stopwords(stop_words_file)


'''
使用TfidfVectorizer()和 CountVectorizer()
分别对数据进行特征的提取，投放到不同的模型中进行实验
'''
#开始使用TF-IDF进行特征的提取，对分词后的中文语句做向量化。
#引进TF-IDF的包
TF_Vec=TfidfVectorizer(max_df=0.8,
                       min_df = 3,
                       stop_words=frozenset(stopwords)
                      )

#拟合数据，将数据准转为标准形式，一般使用在训练集中
train_x_tfvec=TF_Vec.fit_transform(train_x)

#通过中心化和缩放实现标准化，一般使用在测试集中
test_x_tfvec=TF_Vec.transform(test_x)
 
#开始使用CountVectorizer()进行特征的提取。它依据词语出现频率转化向量。并且加入了去除停用词
CT_Vec=CountVectorizer(max_df=0.8,#在超过这一比例的文档中出现的关键词（过于平凡），去除掉。
                       min_df = 3,#在低于这一数量的文档中出现的关键词（过于独特），去除掉。
                       token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b',#使用正则表达式，去除想去除的内容
                       stop_words=frozenset(stopwords))#加入停用词)

#拟合数据，将数据转化为标准形式，一般使用在训练集中
train_x_ctvec=CT_Vec.fit_transform(train_x)

#通过中心化和缩放实现标准化，一般使用在测试集中
test_x_ctvec=CT_Vec.transform(test_x)


'''
使用TF_IDF提取的向量当作数据特征传入模型
'''
#构建模型之前首先将包进行导入
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import time
start_time=time.time()

#创建模型
lr = linear_model.LogisticRegression(penalty='l2', C=1, solver='liblinear', max_iter=1000, multi_class='ovr')

#进行模型的优化，因为一些参数是不确定的
#所以就让模型自己在训练中去确定自己的参数 模型的名字也由LR转变为model
model = GridSearchCV(lr, cv=3, param_grid={
        'C': np.logspace(0, 4, 30),
        'penalty': ['l1', 'l2']
    })

#模型拟合tf-idf拿到的数据
model.fit(train_x_tfvec,train_y)

#查看模型自己拟合的最优参数
print('最优参数：', model.best_params_)

#在训练时查看训练集的准确率
pre_train_y=model.predict(train_x_tfvec)

#在训练集上的正确率
train_accracy=accuracy_score(pre_train_y,train_y)

#训练结束查看预测 输入验证集查看预测
pre_test_y=model.predict(test_x_tfvec)

#查看在测试集上的准确率
test_accracy = accuracy_score(pre_test_y,test_y)
print('使用TF-IDF提取特征使用逻辑回归,让模型自适应参数，进行模型优化\n训练集:{0}\n测试集:{1}'.format(train_accracy,test_accracy))
end_time=time.time()
print("使用模型优化的程序运行时间为",end_time-start_time)

print()
print()

'''
使用ConutVector转化的向量当作特征传入模型
'''
#构建模型之前首先将包进行导入
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import time
start_time=time.time()

#创建模型
lr = linear_model.LogisticRegression(penalty='l2', C=1, solver='liblinear', max_iter=1000, multi_class='ovr')

#进行模型的优化，因为一些参数是不确定的，所以就让模型自己在训练中去确定自己的参数 模型的名字也由LR转变为model
model = GridSearchCV(lr, cv=3, param_grid={
        'C': np.logspace(0, 4, 30),
        'penalty': ['l1', 'l2']
    })

#模型拟合CountVectorizer拿到的数据
model.fit(train_x_ctvec,train_y)

#查看模型自己拟合的最优参数
print('最优参数：', model.best_params_)

#在训练时查看训练集的准却率
pre_train_y=model.predict(train_x_ctvec)

#在训练集上的正确率
train_accracy=accuracy_score(pre_train_y,train_y)

#训练结束查看预测 输入测试集查看预测
pre_test_y=model.predict(test_x_ctvec)

#查看在测试集上的准确率
test_accracy = accuracy_score(pre_test_y,test_y)
print('使用CountVectorizer提取特征使用逻辑回归,让模型自适应参数，进行模型优化\n训练集:{0}\n测试集:{1}'.format(train_accracy,test_accracy))
end_time=time.time()
print("使用模型优化的程序运行时间为",end_time-start_time)

print()
print()

'''使用其他的机器学习模型进行拟合数据进行测试'''
#使用KNN模型
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
start_time=time.time()

#创建模型
Kn = KNeighborsClassifier()

#拟合从tf-idf拿到的数据
Kn.fit(train_x_tfvec,train_y)

#在训练时查看训练集的准确率
pre_train_y=Kn.predict(train_x_tfvec)

#在训练集上的正确率
train_accracy=accuracy_score(pre_train_y,train_y)

#训练结束查看预测 输入测试集查看预测
pre_test_y=Kn.predict(test_x_tfvec)

#查看在测试集上的准确率
test_accracy = accuracy_score(pre_test_y,test_y)
print('使用TfidfVectorizer提取特征使用KNN分类器的准确率\n训练集:{0}\n测试集:{1}'.format(train_accracy,test_accracy))
end_time=time.time()
print("使用KNN分类器的程序运行时间为",end_time-start_time)

print()
print()

### Random Forest Classifier 随机森林分类器 
from sklearn.ensemble import RandomForestClassifier 
import time
start_time=time.time()

#创建模型
Rfc = RandomForestClassifier(n_estimators=8)

#拟合从CounterfVectorizer拿到的数据
Rfc.fit(train_x_ctvec,train_y)

#在训练时查看训练集的准确率
pre_train_y=Rfc.predict(train_x_ctvec)

#在训练集上的正确率
train_accracy=accuracy_score(pre_train_y,train_y)

#训练结束查看预测 输入测试集查看预测
pre_test_y=Rfc.predict(test_x_ctvec)

#查看在测试集上的准确率
test_accracy = accuracy_score(pre_test_y,test_y)
print('使用CounterfVectorizer提取特征使用随机森林分类器的准确率\n训练集:{0}\n测试集:{1}'.format(train_accracy,test_accracy))
end_time=time.time()
print("使用随机森林分类器的程序运行时间为",end_time-start_time)

print()
print()

### Decision Tree Classifier  决策树
from sklearn import tree
import time
start_time=time.time()

#创建模型
Rf = tree.DecisionTreeClassifier()

#拟合从tf-idf拿到的数据
Rf.fit(train_x_tfvec,train_y)

#在训练时查看训练集的准确率
pre_train_y=Rf.predict(train_x_tfvec)

#在训练集上的正确率
train_accracy=accuracy_score(pre_train_y,train_y)

#训练结束查看预测 输入测试集查看预测
pre_test_y=Rf.predict(test_x_tfvec)

#查看在测试集上的准确率
test_accracy = accuracy_score(pre_test_y,test_y)
print('使用tf提取特征使用决策树分类器的准确率\n训练集:{0}\n测试集:{1}'.format(train_accracy,test_accracy))
end_time=time.time()
print("使用决策树分类器的程序运行时间为",end_time-start_time)

print()
print()

### 贝叶斯
from sklearn.naive_bayes import MultinomialNB
import time
start_time=time.time()

#创建模型
Bys = MultinomialNB()

#拟合数据
Bys.fit(train_x_ctvec, train_y)# 学习,拟合模型

#在训练时查看训练集的准确率
pre_train_y=Bys.predict(train_x_ctvec)

#在训练集上的正确率
train_accracy=accuracy_score(pre_train_y,train_y)

#训练结束查看预测 输入测试集查看预测
pre_test_y=Bys.predict(test_x_ctvec)

#查看在测试集上的准确率
test_accracy = accuracy_score(pre_test_y,test_y)
print('使用CounterVectorizer提取特征使用贝叶斯分类器的准确率\n训练集:{0}\n测试集:{1}'.format(train_accracy,test_accracy))
end_time=time.time()
print("使用贝叶斯分类器的程序运行时间为",end_time-start_time)

print()
print()

#使用SVM分类器
from sklearn.svm import SVC
import time
start_time=time.time()

#创建模型
SVM = SVC(C=1.0, kernel='rbf', gamma='auto')

#拟合数据
SVM.fit(train_x_ctvec, train_y)# 学习,拟合模型

#在训练时查看训练集的准确率
pre_train_y=SVM.predict(train_x_ctvec)

#在训练集上的正确率
train_accracy=accuracy_score(pre_train_y,train_y)

#训练结束查看预测 输入测试集查看预测
pre_test_y=SVM.predict(test_x_ctvec)

#查看在测试集上的准确率
test_accracy = accuracy_score(pre_test_y,test_y)
print('使用CounterfVectorizer提取特征使用SVM分类器的准确率\n训练集:{0}\n测试集:{1}'.format(train_accracy,test_accracy))
end_time=time.time()
print("使用SVM分类器的程序运行时间为",end_time-start_time)

