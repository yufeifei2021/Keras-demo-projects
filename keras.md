# Keras

## **1.Keras 中文教程**

Keras 是一个用 Python 编写的高级神经网络 API，它能够以 TensorFlow, CNTK, 或者 Theano 作为后端运行。**Keras 的开发重点是支持快速的实验**。能够以最小的时延把你的想法转换为实验结果，是做好研究的关键。

> **TensorFlow**
>
> 优点：
>
> 1） Google开源的其第二代深度学习技术——被使用在Google搜索、图像识别以及邮箱的深度学习框架。
>
> 2）是一个理想的RNN（递归神经网络）API和实现，**TensorFlow使用了向量运算的符号图方法**，使得新网络的指定变得相当容易，支持快速开发。
>
> 3）TF支持使用ARM/NEON指令实现model decoding（模型解码 ）。
>
> 4）TensorBoard是一个非常好用的网络结构可视化工具，对于分析训练网络非常有用。
>
> 5）编译过程比Theano快，它简单地把符号张量操作映射到已经编译好的函数调用。
>
> 缺点：
>
> 1） 缺点是速度慢，内存占用较大。（比如相对于Torch）
>
> **2）支持的层没有Torch和Theano丰富，特别是没有时间序列的卷积，且卷积也不支持动态输入尺寸，这些功能在NLP中非常有用。**
>
> 当开源TensorFlow后，谷歌立即获得了大量的关注。TensorFlow支持广泛的功能，如图像、手写、语音识别、预测，以及自然语言处理。TensorFlow于2015年11月9日以Apache 2.0许可开源。

> **Torch**
>
> 优点：
>
> 1）Facebook力推的深度学习框架，主要开发语言是C和Lua。
>
> 2）有较好的灵活性和速度。
>
> 3）它实现并且优化了基本的计算单元，使用者可以很简单地在此基础上实现自己的算法，不用浪费精力在计算优化上面。**核心的计算单元使用C或者cuda做了很好的优化。**在此基础之上，使用lua构建了常见的模型。
>
> 4）速度最快。
>
> 5）支持全面的卷积操作：
>
> **\- 时间卷积：输入长度可变，而TF和Theano都不支持，对NLP非常有用。**
>
> \- 3D卷积：Theano支持，TF不支持，对视频识别很有用。
>
> 缺点：
>
> 1）是接口为lua语言，需要一点时间来学习。
>
> **2）没有Python接口。**
>
> 3）与Caffe一样，基于层的网络结构，其扩展性不好，对于新增加的层，需要自己实现（forward, backward and gradient update）。
>
> 4）RNN没有官方支持。
>
> Torch由Facebook的Ronan Collobert和SoumithChintala、Twitter的Clement Farabet（现在在Nvidia），以及Google Deep Mind的KorayKavukcuoglu共同开发。Torch的主要贡献者是Facebook，Twitter和Nvidia。Torch获得BSD 3开源许可。然而，随着Facebook最新宣布其改变航向，使Caffe 2成为主要的深入学习框架，以便在移动设备上部署深入的学习。
>
> **Torch以Lua编程语言实现**。Lua不是主流语言，只有在你的员工熟练掌握之前，才会影响开发人员的整体效率。
>
> **Torch缺乏TensorFlow的分布式应用程序管理框架或者在MXNet或Deeplearning4J中支持YARN。缺乏大量的API编程语言也限制了开发人员。**

> **CNTK**
>
> CNTK是一种深度神经网络，最初是为了提高语音识别而开发的。**CNTK支持RNN和CNN类型的神经模型，使其成为处理图像、手写和语音识别问题的最佳候选。**CNTK支持使用Python或C++编程接口的64位Linux和Windows操作系统，并根据MIT许可发布。
>
> CNTK与TensorFlow和Theano的组成相似，其网络被指定为向量运算的符号图，如矩阵的加法/乘法或卷积。此外，像TensorFlow和Theano一样，CNTK允许构建网络层的细粒度。构建块（操作）的细粒度允许用户创造新的复合层类型，而不用低级语言实现（如Caffe）。
>
> 像Caffe一样，CNTK也是基于C++的、具有跨平台的CPU/GPU支持。CNTK在Azure GPU Lab提供了最高效的分布式计算性能。目前，**CNTK对ARM架构的缺乏支持，限制了其在移动设备上的功能。**
>
> **ARM 架构**，过去称作进阶精简指令集机器，是一个 32 位精简指令集 (RISC) 处理器架构，被广泛地使用在嵌入式系统设计。

> **Theano**
>
> 优点：
>
> 1）2008年诞生于蒙特利尔理工学院，**主要开发语言是Python**。
>
> 2）Theano派生出了大量深度学习Python软件包，最著名的包括Blocks和Keras。
>
> 3）Theano的最大特点是非常的灵活，适合做学术研究的实验，且对递归网络和语言建模有较好的支持。
>
> 4）是第一个使用符号张量图描述模型的架构。
>
> 5）支持更多的平台。
>
> 6）在其上有可用的高级工具：Blocks, Keras等。
>
> 缺点：
>
> 1）编译过程慢，但同样采用符号张量图的TF无此问题。
>
> 2）import theano也很慢，它导入时有很多事要做。
>
> 3）作为开发者，很难进行改进，因为code base是Python，而C/CUDA代码被打包在Python字符串中。
>
> Theano由蒙特利尔大学学习算法学院（MILA）积极维护。以Theano的创始人YoshuaBengio为首，该实验室拥有约30-40名教师和学生，是深度学习研究的重要贡献者。Theano支持快速开发高效的机器学习算法，并通过BSD许可发布。
>
> Theano架构相当简单，整个代码库和接口是Python，其中C/CUDA代码被打包成Python字符串。对一个开发者来说这很难驾驭、调试和重构。
>
> Theano开创了使用符号图来编程网络的趋势。Theano的符号API支持循环控制，即所谓的扫描，这使得实现RNN更容易、更高效。
>
> Theano缺乏分布式应用程序管理框架，只支持一种编程开发语言。Theano是学术研究的一个很好的工具，在一个CPU上比TensorFlow更有效地运行。然而，在开发和支持大型分布式应用程序时，可能会遇到挑战。

如果你在以下情况下需要深度学习库，请使用 Keras：

- 允许简单而快速的原型设计（由于用户友好，高度模块化，可扩展性）。
- **同时支持**卷积神经网络和循环神经网络，以及两者的组合。
- **在 CPU 和 GPU 上无缝运行**。

Keras 兼容的 Python 版本: Python 2.7-3.6。

**基于 Python 实现。 Keras 没有特定格式的单独配置文件。模型定义在 Python 代码中，这些代码紧凑，易于调试，并且易于扩展。**

**Keras 的核心数据结构是 model，一种组织网络层的方式。最简单的模型是 Sequential 顺序模型，它由多个网络层线性堆叠。对于更复杂的结构，你应该使用 Keras 函数式 API，它允许构建任意的神经网络图。**

> Sequential模型
>
> 字面上的翻译是顺序模型，给人的第一感觉是那种简单的线性模型，但实际上Sequential模型可以构建非常复杂的神经网络，包括全连接神经网络、卷积神经网络 (CNN)、循环神经网络 (RNN)等等。 这里Sequential更准确的应该理解为**堆叠，通过堆叠多层，构建出深度神经网络。 Sequential模型的核心是添加layers（图层），**以下展示如何将一些最流行的图层添加到模型中： 从我们所学习到的机器学习知识可以知道，**机器学习通常包括定义模型、定义优化目标、输入数据、训练模型，最后通常还需要使用测试数据评估模型的性能。**

Sequential 模型如下所示：

```python
from keras.models import Sequential

model = Sequential()
```

可以简单地**使用 .add() 来堆叠模型**：

```python
from keras.layers import Dense

model.add(Dense(units=64, activation='relu', input_dim=100))//单位 激活函数 数据的维度
model.add(Dense(units=10, activation='softmax'))
```

在完成了模型的构建后, 可以**使用 .compile() 来配置学习过程**：

```python
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```

> categorical_crossentropy（交叉熵损失函数) 交叉熵是用来评估当前训练得到的概率分布与真实分布的差异情况。 它刻画的是实际输出（概率）与期望输出（概率）的距离，也就是交叉熵的值越小，两个概率分布就越接近。
>
> categorical_crossentropy主要用于目标结果是独热编码(one-hot encoding)

> SGD(随机梯度下降)      监督学习
>
>  随机梯度下降在计算下降最快的方向时**随机选一个数据进行计算**，而不是扫描全部训练数据集，**这样就加快了迭代速度**。 随机梯度下降并不是沿着J(θ)下降最快的方向收敛，而是震荡的方式趋向极小点。

如果需要，你还可以**进一步地配置你的优化器**。Keras 的核心原则是使事情变得相当简单，同时又允许用户在需要的时候能够进行完全的控制（终极的控制是源代码的易扩展性）。

```python
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
                                             //学习率 动量 加速
```

现在，你可以批量地在训练数据上进行迭代了：

```python
# x_train 和 y_train 是 Numpy 数组 -- 就像在 Scikit-Learn API 中一样。
model.fit(x_train, y_train, epochs=5, batch_size=32)//自动！
```

> model.fit()方法用于执行训练过程
>
>  model.fit( 训练集的输入特征，
>
> ​             训练集的标签，  
>
> ​             batch_size,  #每一个batch的大小
>
> ​             epochs,   #迭代次数
>
> ​             validation_data = (测试集的输入特征，测试集的标签），
>
> ​             validation_split = 从测试集中划分多少比例给训练集，
>
> ​             validation_freq = 测试的epoch间隔数）

或者，你可以手动地将批次的数据提供给模型：

```python
model.train_on_batch(x_batch, y_batch)//手动！
```

**只需一行代码就能评估模型性能**：

```python
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
```

或者对新的数据生成预测：

```python
classes = model.predict(x_test, batch_size=128)
```

**构建一个问答系统**，一个图像分类模型，一个神经图灵机，或者其他的任何模型，就是这么的快。深度学习背后的思想很简单，那么它们的实现又何必要那么痛苦呢？



在安装 Keras 之前，请安装以下后端引擎之一：TensorFlow，Theano，或者 CNTK。我们推荐 TensorFlow 后端。TensorFlow 安装指引:https://tensorflow.google.cn/install

TensorFlow 安装指引:https://tensorflow.google.cn/install

- cuDNN (如果你计划在 GPU 上运行 Keras，建议安装)。
- HDF5 和 h5py (如果你需要将 Keras 模型保存到磁盘，则需要这些)。
- graphviz 和 pydot (用于可视化工具绘制模型图)。
  然后你就可以安装 Keras 本身了。有两种方法安装 Keras：

使用 PyPI 安装 Keras (推荐)：

```
sudo pip install keras
```

如果你使用 virtualenv 虚拟环境, 你可以避免使用 sudo：

```
pip install keras
```

或者：使用 GitHub 源码安装 Keras：
首先，使用 git 来克隆 Keras：

```
git clone https://github.com/keras-team/keras.git
```

然后，cd 到 Keras 目录并且运行安装命令：

```
cd keras
sudo python setup.py install
```

配置你的 Keras 后端
默认情况下，Keras 将使用 TensorFlow 作为其张量操作库。请跟随这些指引来配置其他 Keras 后端。

## **2.Keras 图像增强**

使用神经网络和[深度学习模型](https://geek-docs.com/deep-learning/deep-learning-tutorial/waht-is-deep-learning.html)时**需要准备数据**。对于复杂的图像对象物件识别的任务，也需要越来越多的数据增强功能，让模型的效能更好。

### 2.1Keras图像增强API

Keras提供**ImageDataGenerator类别，用于定义图像数据准备和增强的配置**。

这包括以下功能：

- 图像随机旋转(rotate)

- 图像随机偏移(shift)

- 图像随机推移错切(shear)

- 图像随机翻转(flip)

- Sample-wise 图像像素标准化

- Feature-wise 图像像素标准化

- ZCA 白化转换

- 图像张量维度的重排序

- 储存增强图像数据

  

  第一步：

  **增强图像生成器**可以透过以下的方法创建：

```python
datagen = ImageDataGenerator()
```

**Keras**并不是在记忆体中对整个图像数据集执行图像转换操作，而**是设计为通过深度学习模型训练过程进行迭代**，从而动态地创建增强的图像数据。这会减少记忆体开销，但在模型训练期间会增加一些额外的时间成本。



​	 第二步：

创建并配置好ImageDataGenerator之后，必须**用数据来训练它**。这个步骤将计算实际执行转换到图像数据所需的任何统计参数。通过调用数据生成器上的**fit( )函数**并将传递您的训练数据集来完成这个前置动作。

```python
datagen.fit(train) //fit传递训练数据集
```

​	

​	第三步：

**数据生成器本身实际上是一个迭代器**，当被呼叫时返回一个批量的图像资料。我们可以通过调用**flow( )函数**来配置批量大小并并获取批量的图像资料。

```python
X_batch, y_batch = datagen.flow(train, train, batch_size=32) 
							//flow配置批量大小并获取
```



​	第四步：

最后我们就可以**使用数据生成器**。我们不必调用我们模型的fit( )函数，而是调用**fit_generator( )函数**，并传递数据生成器的实例(instance)和每个循环的步数(steps_per_epoch)以及要训练的循环总数(epochs)。

```python
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size), //配置、获取批量
                    steps_per_epoch=len(X_train)/batch_size, //循坏的步数
                    epochs=nb_epoch)  //循坏总数
```

### 2.2图像增强相关参数的比较

这些例子中我们将使用**MNIST手写数字识别的图像集**。

首先，看看训练数据集中的前6个图像。

```python
import  keras 
# Plot images 
from  keras.datasets  import  mnist 
import  matplotlib.pyplot  as  plt

# load data 
( X_train ,  y_train ),  ( X_test ,  y_test )  =  mnist . load_data ()

plt . figure ( figsize = ( 8 , 8 ))  #设定每个图像显示的大小

#产生一个3x2网格的组合图像
for  i  in  range ( 0 ,  6 ):  //表示前6个图像
    plt . subplot ( 330 + 1 + i )  # (331) ->第一个子图像, (332) ->第二个子图像
    plt . title ( y_train [ i ]) 
    plt . axis ( 'off' )      #不显示坐标
    plt . imshow ( X_train [ i ],  cmap = plt . get_cmap ( 'gray' )) # 以灰阶的图像显示

#展现出图像
plt . show ()
```

![在Keras中使用图像增强来进行深度学习](https://img.geek-docs.com/keras/keras-examples/image-enhance-1.png)

运行这个例子会提供了以上的图片，我们可以用它来**比较下面例子中的图片准备和增强对图像转换效果的理解**。

### 2.3随机旋转

有时样本数据中的图像可能在场景中会有不同的旋转的角度。

在训练模型时，通过随机旋转训练数据集里的图像来产生一些新的训练图像来帮助模型更好地处理图像的旋转问题。

下面的范例通过设置**rotation_range参数**来创建更多的MNIST数字图像**(随机旋转最多可达90度)**。

```python
#随机旋转(Random Rotations) 

from  keras.datasets  import  mnist 
from  keras.preprocessing.image  import  ImageDataGenerator 
import  matplotlib.pyplot  as  plt

#载入数据
( X_train ,  y_train ),  ( X_test ,  y_test )  =  mnist . load_data ()

#将图像数据集的维度进行改变
#改变前: [样本数,图像宽,图像高] ->改变后: [样本数,图像宽,图像高,图像频道数] 
X_train  =  X_train . reshape ( X_train . shape [ 0 ],  28 ,  28 ,  1 ) 
X_test   =  X_test . reshape ( X_test . shape [ 0 ],  28 ,  28 ,  1 )

#将像素值由"整数(0~255)"换成"浮点数(0.0~255.0)" 
X_train  =  X_train . astype ( 'float32' ) 
X_test  =  X_test . astype ( 'float32' )

#定义"图像数据增强产生器"的参数
datagen  =  ImageDataGenerator ( rotation_range = 90 )

#透过训练数据集来训练(fit)图像数据增强产生器的实例
datagen . fit ( X_train )

#设定要"图像数据增强产生器"产生的图像批次值(batch size) 
# "图像数据增强产生器"会根据设定回传指定批次量的新生成图像数据
for  X_batch ,  y_batch  in  datagen . flow ( X_train ,  y_train ,  batch_size = 9 ): 
    plt . figure ( figsize = ( 8 , 8 ))  #设定每个图像显示的大小
    #产生一个3x3网格的组合图像
    for  i  in  range ( 0 ,  6 ): 
        plt . subplot ( 331+ i ) 
        plt . title ( y_batch [ i ])  #秀出图像的真实值
        plt . axis ( 'off' )      #不显示坐标
        plt . imshow ( X_batch [ i ] . reshape ( 28 , 28 ),  cmap = plt . get_cmap ( 'gray' ))

    plt . show () 
    break  #跳出回圈
```

![在Keras中使用图像增强来进行深度学习](https://img.geek-docs.com/keras/keras-examples/image-enhance-2.png)

运行以上范例，可以看到图像已左右旋转至90度的极限。**这个”随机旋转”的功能对解决MNIST数字辨识这个问题没有什么帮助**，因为**MNIST数字有一个标准化的方向**，但是当您要辨识的对象或物件需要不同方向的照片来进行训练学习时，这种转换可能有相当的帮助。

> 辨识MNIST数据集没有什么帮助，但是日常生活中，表示照片还是有相当大的帮助的。

### 2.4随机偏移

图像中的对象或物件在图框中的位置可能在最中间，他们可能以各种不同的方式**偏离图框的中心点**。

可以训练深度学习网络，通过**人工创建**训练数据的偏移图像让模型能够辨识和处理偏离中心的对象或物件。Keras的图像数据增强产生器支持通过**width_shift_range**和**height_shift_range**两个参数设定来对训练数据进行**水平**和**垂直**随机转换来产生新图像数据。

```python
#随机偏移(Random Shifts) 
from  keras.datasets  import  mnist 
from  keras.preprocessing.image  import  ImageDataGenerator 
import  matplotlib.pyplot  as  plt

#载入数据
( X_train ,  y_train ),  ( X_test ,  y_test )  =  mnist . load_data ()

#将图像数据集的维度进行改变
#改变前: [样本数,图像宽,图像高] ->改变后: [样本数,图像宽,图像高,图像频道数] 
X_train  =  X_train . reshape ( X_train . shape [ 0 ],  28 ,  28 ,  1 ) 
X_test   =  X_test . reshape ( X_test . shape [ 0 ],  28 ,  28 ,  1 )

#将像素值由"整数(0~255)"换成"浮点数(0.0~255.0)" 
X_train  =  X_train . astype ( 'float32' ) 
X_test  =  X_test . astype ( 'float32' )

#定义"图像数据增强产生器(ImageDataGenerator)"的参数
shift  =  0.2 
datagen  =  ImageDataGenerator ( width_shift_range = shift ,  height_shift_range = shift )

#透过训练数据集来训练(fit)图像数据增强产生器(ImageDataGenerator)的实例
datagen . fit ( X_train )

#设定要"图像数据增强产生器(ImageDataGenerator)"产生的图像批次值(batch size) 
# "图像数据增强产生器(ImageDataGenerator)"会根据设定回传指定批次量的新生成图像数据
for  X_batch ,  y_batch  in  datagen . flow ( X_train ,  y_train ,  batch_size = 9 ): 
    plt . figure ( figsize = ( 8 , 8 ))  #设定每个图像显示的大小
    #产生一个3x3网格的组合图像
    for  i  in  range ( 0 ,  6 ): 
        plt . subplot ( 331+ i ) 
        plt . title ( y_batch [ i ])  #秀出图像的真实值
        plt . axis ( 'off' )      #不显示坐标
        plt . imshow ( X_batch [ i ] . reshape ( 28 , 28 ),  cmap = plt . get_cmap ( 'gray' ))

    plt . show () 
    break  #跳出回圈
```

![在Keras中使用图像增强来进行深度学习](https://img.geek-docs.com/keras/keras-examples/image-enhance-3.png)

运行此范例将创建新的MNIST数字图像的位移版本。这对于MNIST来说不是必须的，因为手写数字已经居中，但是你可以看到这对于更复杂的图像问题来说可能是有用的。

> 辨识MNIST数据集没有什么帮助，因为MNIST数据集中的手写数字已经居中。
>
> 日常生活中，面对复杂的图像还是有相当大的帮助的。

### 2.5随机推移错切

透过一个图像的错切变换，图像以它的**中心垂直轴不变动的方式变形**。可以用这样的技术来产生新的经过错切变换的图像数据。

![在Keras中使用图像增强来进行深度学习](https://img.geek-docs.com/keras/keras-examples/330px-Eigen.jpg)



Keras的”图像数据增强产生器”支持使用**shear_range参数**设定来对训练数据进行**图像错切变换**来产生新图像数据。

```python
#随机推移错切(Random Shear) 
from  keras.datasets  import  mnist 
from  keras.preprocessing.image  import  ImageDataGenerator 
import  matplotlib.pyplot  as  plt

#载入数据
( X_train ,  y_train ),  ( X_test ,  y_test )  =  mnist . load_data ()

#将图像数据集的维度进行改变
#改变前: [样本数,图像宽,图像高] ->改变后: [样本数,图像宽,图像高,图像频道数] 
X_train  =  X_train . reshape ( X_train . shape [ 0 ],  28 ,  28 ,  1 ) 
X_test   =  X_test . reshape ( X_test . shape [ 0 ],  28 ,  28 ,  1 )

#将像素值由"整数(0~255)"换成"浮点数(0.0~255.0)" 
X_train  =  X_train . astype ( 'float32' ) 
X_test  =  X_test . astype ( 'float32' )

#定义"图像数据增强产生器(ImageDataGenerator)"的参数
shear_range = 1.25  #推移错切的强度
datagen  =  ImageDataGenerator ( shear_range = shear_range )

#透过训练数据集来训练(fit)图像数据增强产生器(ImageDataGenerator)的实例
datagen . fit ( X_train )

#设定要"图像数据增强产生器(ImageDataGenerator)"产生的图像批次值(batch size) 
# "图像数据增强产生器(ImageDataGenerator)"会根据设定回传指定批次量的新生成图像数据
for  X_batch ,  y_batch  in  datagen . flow ( X_train ,  y_train ,  batch_size = 9 ): 
    plt . figure ( figsize = ( 8 , 8 ))  #设定每个图像显示的大小
    #产生一个3x3网格的组合图像
    for  i  in  range ( 0 ,  6 ): 
        plt . subplot ( 331+ i ) 
        plt . title ( y_batch [ i ])  #秀出图像的真实值
        plt . axis ( 'off' )      #不显示坐标
        plt . imshow ( X_batch [ i ] . reshape ( 28 , 28 ),  cmap = plt . get_cmap ( 'gray' ))

    plt . show () 
    break  #跳出回圈
```

![在Keras中使用图像增强来进行深度学习](https://img.geek-docs.com/keras/keras-examples/image-enhance-4.png)



### 2.6随机镜像翻转

另一个可以**提高大型复杂图像问题辨识性能的图像增强法**，是在图像训练数据中**利用图像随机翻转**来产生新的图像数据。

Keras的”图像数据增强产生器”支持使用**vertical_flip**和**horizontal_flip**两个参数设定来对训练数据进行**垂直**和**水平**轴随机翻转来产生新图像数据。

```python
#随机镜像翻转(Random Flips) 
from  keras.datasets  import  mnist 
from  keras.preprocessing.image  import  ImageDataGenerator 
import  matplotlib.pyplot  as  plt

#载入数据
( X_train ,  y_train ),  ( X_test ,  y_test )  =  mnist . load_data ()

#将图像数据集的维度进行改变
#改变前: [样本数,图像宽,图像高] ->改变后: [样本数,图像宽,图像高,图像频道数] 
X_train  =  X_train . reshape ( X_train . shape [ 0 ],  28 ,  28 ,  1 ) 
X_test   =  X_test . reshape ( X_test . shape [ 0 ],  28 ,  28 ,  1 )

#将像素值由"整数(0~255)"换成"浮点数(0.0~255.0)" 
X_train  =  X_train . astype ( 'float32' ) 
X_test  =  X_test . astype ( 'float32' )

#定义"图像数据增强产生器(ImageDataGenerator)"的参数
datagen  =  ImageDataGenerator ( horizontal_flip = True ,  vertical_flip = True )

#透过训练数据集来训练(fit)图像数据增强产生器(ImageDataGenerator)的实例
datagen . fit ( X_train )

#设定要"图像数据增强产生器(ImageDataGenerator)"产生的图像批次值(batch size) 
# "图像数据增强产生器(ImageDataGenerator)"会根据设定回传指定批次量的新生成图像数据
for  X_batch ,  y_batch  in  datagen . flow ( X_train ,  y_train ,  batch_size = 9 ): 
    plt . figure ( figsize = ( 8 , 8 ))  #设定每个图像显示的大小
    #产生一个3x3网格的组合图像
    for  i  in  range ( 0 ,  6 ): 
        plt . subplot ( 331+ i ) 
        plt . title ( y_batch [ i ])  #秀出图像的真实值
        plt . axis ( 'off' )      #不显示坐标
        plt . imshow ( X_batch [ i ] . reshape ( 28 , 28 ),  cmap = plt . get_cmap ( 'gray' ))

    plt . show () 
    break  #跳出回圈
```

![在Keras中使用图像增强来进行深度学习](https://img.geek-docs.com/keras/keras-examples/image-enhance-5.png)

运行这个范例，您可以看到翻转的数字。翻转数字对于MNIST来说是没有用的，因为它们总是具有正确的左右方向，但是这对于可能具有不同方向的场景中的对象的照片的问题可能是有用的。

> 辨识MNIST数据集没有什么帮助，因为MNIST数据集中的手写数字总是具有正确的左右方向。
>
> 日常生活中，面对具有不同场景中的对象照片时还是有相当大的帮助的。

### 2.7影像特征标准化

把整个数据集中的**每个像素值来进行标准化**(standardize pixel)。这被称为**特征标准化**，影像特征标准化的程序如同我们对一般表格数据集中每个列会进次的资料标准化。

通过在ImageDataGenerator类别上设置**featurewise_center和featurewise_std_normalization的参数**来执行影像特征标准化。

```python
#对整个资料集进行"影像特征标准化" , mean=0, stdev=1 
from  keras.datasets  import  mnist 
from  keras.preprocessing.image  import  ImageDataGenerator 
import  matplotlib.pyplot  as  plt

#在keras的backend预设使用tensorflow 
#而tensorflow在处理图像是的资料结构是3维的张量(图像宽:width,图像高:height,图像频道:channel)

#载入数据
( X_train ,  y_train ),  ( X_test ,  y_test )  =  mnist . load_data ()

#将图像数据集的维度进行改变
#改变前: [样本数,图像宽,图像高] ->改变后: [样本数,图像宽,图像高,图像频道数] 
X_train  =  X_train . reshape ( X_train . shape [ 0 ],  28 ,  28 ,  1 ) 
X_test   =  X_test . reshape ( X_test . shape [ 0 ],  28 ,  28 ,  1 )

#将像素值由"整数(0~255)"换成"浮点数(0.0~255.0)" 
X_train  =  X_train . astype ( 'float32' ) 
X_test  =  X_test . astype ( 'float32' )

#定义"图像数据增强产生器(ImageDataGenerator)"的参数
datagen  =  ImageDataGenerator ( featurewise_center = True ,  
                             featurewise_std_normalization = True )

#透过训练数据集来训练(fit)图像数据增强产生器(ImageDataGenerator)的实例
datagen . fit ( X_train )

#设定要"图像数据增强产生器(ImageDataGenerator)"产生的图像批次值(batch size) 
# "图像数据增强产生器(ImageDataGenerator)"会根据设定回传指定批次量的新生成图像数据
for  X_batch ,  y_batch  in  datagen . flow ( X_train ,  y_train ,  batch_size = 9 ): 
    plt . figure ( figsize = ( 8 , 8 ))  #设定每个图像显示的大小
    #产生一个3x3网格的组合图像
    for  i  in  range ( 0 ,  6 ): 
        plt . subplot ( 331+ i ) 
        plt . title ( y_batch [ i ])  #秀出图像的真实值
        plt . axis ( 'off' )      #不显示坐标
        plt . imshow ( X_batch [ i ] . reshape ( 28 , 28 ),  cmap = plt . get_cmap ( 'gray' ))

    plt . show () 
    break  #跳出回圈
```

![在Keras中使用图像增强来进行深度学习](https://img.geek-docs.com/keras/keras-examples/image-enhance-6.png)



### 2.7 ZCA白化转换

图像的白化变换是**为了减少图像像素矩阵中的冗余/去除相关性(共线性)的线性特征所进行的代数运算转换**。

进行图像中像素矩阵去除相关性的转换是**为了更好地突显图像中的结构和特征到学习演算法**。

> 这个转换的概念在以前大多是**使用主成分分析（PCA）技术来执行图像白化**。
>
> **与PCA是不同的是**，ZCA的技术显示了更好的结果，对图像进行**ZCA转换**后的图像可以**保留原始尺寸的大小**，从而使转换后的图像仍然看起来像原始图像。



您可以通过将**zca_whitening参数**设置为True来执行ZCA白化转换。

```python
# ZCA白化转换(ZCA Whitening) 
from  keras.datasets  import  mnist 
from  keras.preprocessing.image  import  ImageDataGenerator 
import  matplotlib.pyplot  as  plt

#载入数据
( X_train ,  y_train ),  ( X_test ,  y_test )  =  mnist . load_data ()

#将图像数据集的维度进行改变
#改变前: [样本数,图像宽,图像高] ->改变后: [样本数,图像宽,图像高,图像频道数] 
X_train  =  X_train . reshape ( X_train . shape [ 0 ],  28 ,  28 ,  1 ) 
X_test   =  X_test . reshape ( X_test . shape [ 0 ],  28 ,  28 ,  1 )

#将像素值由"整数(0~255)"换成"浮点数(0.0~255.0)" 
X_train  =  X_train . astype ( 'float32' ) 
X_test  =  X_test . astype ( 'float32' )

#定义"图像数据增强产生器(ImageDataGenerator)"的参数
datagen  =  ImageDataGenerator ( zca_whitening = True )

#透过训练数据集来训练(fit)图像数据增强产生器(ImageDataGenerator)的实例
datagen . fit ( X_train )

#设定要"图像数据增强产生器(ImageDataGenerator)"产生的图像批次值(batch size) 
# "图像数据增强产生器(ImageDataGenerator)"会根据设定回传指定批次量的新生成图像数据
for  X_batch ,  y_batch  in  datagen . flow ( X_train ,  y_train ,  batch_size = 9 ): 
    plt . figure ( figsize = ( 8 , 8 ))  #设定每个图像显示的大小
    #产生一个3x3网格的组合图像
    for  i  in  range ( 0 ,  6 ): 
        plt . subplot ( 331+ i ) 
        plt . title ( y_batch [ i ])  #秀出图像的真实值
        plt . axis ( 'off' )      #不显示坐标
        plt . imshow ( X_batch [ i ] . reshape ( 28 , 28 ),  cmap = plt . get_cmap ( 'gray' ))

    plt . show () 
    break  #跳出回圈
```

![在Keras中使用图像增强来进行深度学习](https://img.geek-docs.com/keras/keras-examples/image-enhance-7.png)

运行以上范例，您可以在图像中看到相同的一般数字结构以及每个数字的轮廓如何被突显。

> ZCA白化就是进行完PCA白化后，把数据“旋转回去”。



### 2.8储存增强的图像数据

> 数据准备和增强由Keras在模型训练的运行中进行。
>
> 这对记忆体的使用效率很高，但有可能需要在训练期间使用在档案系统中图像档。例如，也许希望稍后使用不同的套件进行分析，或者只生成一次新增强的图像数据，然后在多种不同的深度学习模型或配置中使用它们。

**Keras允许保存训练期间动态生成的图像**。可以在训练前指定给flow（）函数有关于想保存的目录、文件名前缀和图像文件类型。然后在训练过程中，动态生成的图像将被写入到档案系统中。

下面的范例演示了这一点，并将6个图像写入“images”子目录，前缀为“aug”，文件类型为PNG。

```python
#储存增强的图像数据(Saving Augmented Images to File) 
from  keras.datasets  import  mnist 
from  keras.preprocessing.image  import  ImageDataGenerator 
import  matplotlib.pyplot  as  plt 
import  os

#载入数据
( X_train ,  y_train ),  ( X_test ,  y_test )  =  mnist . load_data ()

#将图像数据集的维度进行改变
#改变前: [样本数,图像宽,图像高] ->改变后: [样本数,图像宽,图像高,图像频道数] 
X_train  =  X_train . reshape ( X_train . shape [ 0 ],  28 ,  28 ,  1 ) 
X_test   =  X_test . reshape ( X_test . shape [ 0 ],  28 ,  28 ,  1 )

#将像素值由"整数(0~255)"换成"浮点数(0.0~255.0)" 
X_train  =  X_train . astype ( 'float32' ) 
X_test  =  X_test . astype ( 'float32' )

#定义"图像数据增强产生器(ImageDataGenerator)"的参数
datagen  =  ImageDataGenerator ()

#透过训练数据集来训练(fit)图像数据增强产生器(ImageDataGenerator)的实例
datagen . fit ( X_train )

#产生要保存图像档案的目录
if  not  os . path . exists ( 'images' ): 
    os . makedirs ( 'images' )

#设定要"图像数据增强产生器(ImageDataGenerator)"产生的图像批次值(batch size) 
# "图像数据增强产生器(ImageDataGenerator)"会根据设定回传指定批次量的新生成图像数据
for  X_batch ,  y_batch  in  datagen . flow ( X_train ,  y_train ,  batch_size = 9 ,  save_to_dir = 'images' ,  save_prefix = 'aug' ,  save_format = 'png' ): 
    plt . figure ( figsize = ( 8 , 8 ))  #设定每个图像显示的大小
    #产生一个3x3网格的组合图像
    for  i  in  range ( 0 ,  6 ): 
        plt . subplot ( 331 + i ) 
        plt . title ( y_batch [ i ])  #秀出图像的真实值
        plt . axis ( 'off' )      #不显示坐标
        plt . imshow ( X_batch [ i ] . reshape ( 28 , 28 ),  cmap = plt . get_cmap ('gray' ))

    plt . show () 
    break  #跳出回圈
```

![在Keras中使用图像增强来进行深度学习](https://img.geek-docs.com/keras/keras-examples/image-enhance-8.png)

运行范例，可以看到**只有在图像生成时才会也把写入图像写到档案系统中**。

### 2.9用Keras增强图像数据的技巧

图像数据很独特，因为可以查看数据和数据的转换副本，并快速了解模型如何通过您的图像感知到某种影像特征。

以下是从图像数据准备和增强深度学习中获得最多的一些时间。

- 查看图像数据集花一些时间仔细检查你的数据集。看图像。注意可能有利于模型训练过程的图像准备和增强，例如需要处理场景中不同的轮班，旋转或翻转对象。
- 查看增强过的图像增强操作完成后，查看一些范例图像。理智地知道你正在使用什么样的图像变换是一回事，真正看到范例图像是非常不同的事情。查看您正在使用的单个增强图像以及您计划使用的全套增强图像。您可能会看到简化或进一步增强您的模型训练的方法。
- 评估一整套变换尝试多个图像数据准备和增强方案。通常情况下，您认为这样做不会有好处的转换,最后您可能会对数据增强后对模型的结果感到惊讶。

### 2.10总结

> 了解图像数据准备和增强的一些面向。

发现一系列可以轻松使用Keras来使用图像数据增强(data augmentation)应用在深度学习模型的技术：

- Keras中的**ImageDataGenerator API**可以用来**动态产生新的转换后的图像来用于训练**。
- 图像像素的标准化手法。
- ZCA白化转换。
- 对图像进行随机旋转，移动和翻转。
- 如何将转换后的图像保存到档案系统中以备后用。

## 3.Keras 函数式API

Keras使得创建深度学习模型变得快速而简单。

1.**序贯(sequential)API**

允许为大多数问题逐层堆叠创建模型。虽然说对很多的应用来说, 这样的一个手法很简单也解决了很多深度学习网络结构的构建，但是也有限制---**不允许你创建模型有共享层或有多个输入或输出的网络**。

2.Keras中的**函数式(functional)API**

创建网络模型的另一种方式，它提供了更多的灵活性，包括创建更复杂的模型。

> Keras的API：序贯API、函数式API

在这个文章中，了解**如何使用Keras函数式API进行深度学习**。



完成这个文章的相关范例， 您将知道：

- Sequential和Functional API之间的**区别**。
- 如何使用功能性API定义简单的多层感知器(MLP)，卷积神经网络(CNN)和递归神经网络(RNN)模型。
- 如何用共享层和多个输入输出来定义更复杂的模型。
  ![如何使用Keras函数式API进行深度学习](https://img.geek-docs.com/keras/keras-examples/Popular-Neural-Network-Architecture.jpg)

```python
#这个Jupyter Notebook的环境
import  platform 
import  tensorflow 
import  keras 
print ( "Platform: {} " . format ( platform . platform ())) 
print ( "Tensorflow version: {} " . format ( tensorflow . __version__ )) 
print ( " Keras version: {} " . format ( keras . __version__ ))

% matplotlib inline
 import  matplotlib.pyplot  as  plt 
import  matplotlib.image  as  mpimg 
import  numpy  as  np 
from  IPython.display  import  Image
```

> Platform: Windows-7-6.1.7601-SP1
> Tensorflow version: 1.4.0
> Keras version: 2.1.1



### 3.1Keras序贯模型(Sequential Models)

Keras提供了一个**Sequential模型API**。

它是创建深度学习模型的一种**相对简单**的方法，我们透过创建Kears的Sequential类别实例(instance), 然后创建模型图层并添加到其中。

例如，可以定义多个图层并将**以阵列的方式**一次做为参数传递给Sequential：

```python
from  keras.models  import  Sequential 
from  keras.layers  import  Dense

#构建模型
model  =  Sequential ([ Dense ( 2 ,  input_shape = ( 1 ,)),  Dense ( 1 )])
```

当然我们也可以**一层一层也分段添加**上去：

```python
from  keras.models  import  Sequential 
from  keras.layers  import  Dense

#构建模型
model  =  Sequential () 
model . add ( Dense ( 2 ,  input_shape = ( 1 ,))) 
model . add ( Dense ( 1 ))
```

Sequential模型API对于在大多数情况下非常有用与方便，但也有一些局限性。例如，网络拓扑结构可能具有多个不同输入，产生多个输出或重复使用共享图层的复杂模型。

### 3.2Keras函数式(functional)API构建模型

Keras函数式API为构建网络模型提供了**更为灵活**的方式。

允许定义多个输入或输出模型以及共享图层的模型；允许定义动态(ad-hoc)的非周期性(acyclic)网络图。

**模型是通过创建层的实例(layer instances)并将它们直接相互连接成对来定义的**，然后定义一个模型(model)来指定那些层是要作为这个模型的输入和输出。

依次看看Keras功能API的三个独特特性：

#### 3.2.1定义输入

与Sequential模型不同，必须创建**独立**的Input层物件的instance并定义输入数据张量的维度形状。

**输入层**采用一个张量形状参数，它是一个tuple，用于宣吿输入张量的维度。

例如: 我们要把MNIST的每张图像(28×28)打平成一个一维(784)的张量做为一个多层感知器(MLP)的Input

```python
from  keras.layers  import  Input

mnist_input  =  Input ( shape = ( 784 ,))
```

#### 3.2.2连接不同的网络层

**模型中的神经层是成对连接的**，通过在定义每个新神经层时指定输入的来源来完成的。使用**括号表示法**，以便在创建图层之后，指定作为输入的神经层。

我们可以像上面那样创建输入层，然后创建一个隐藏层作为密集层，它接收来自输入层的输入。

```python
from  keras.layers  import  Input 
from  keras.layers  import  Dense

mnist_input  =  Input ( shape = ( 784 ,)) 
hidden  =  Dense ( 512 )( mnist_input )
```

正是这种**逐层连接的方式**赋予功能性API灵活性。您可以看到开始一些动态的神经网络是多么容易。

#### 3.2.3创建模型

在创建所有模型图层并将它们连接在一起之后，您必须**定义一个模型(Model)物件的instance**。

与Sequential API一样，**这个模型可以用于总结，拟合(fit)，评估(evaluate)和预测**。

Keras提供了一个Model类别，您可以使用它从创建的图层创建模型的instance。它会要求您只指定整个模型的第一个输入层和最后一个的输出层。例如：

```python
from  keras.layers  import  Input 
from  keras.layers  import  Dense 
from  keras.models  import  Model

mnist_input  =  Input ( shape = ( 784 ,)) 
hidden  =  Dense ( 512 )( mnist_input )

model  =  Model ( inputs = mnist_input ,  outputs = hidden )
```

> 现在我们已经知道了Keras函数式API的**所有关键部分**，让我们通过定义一系列不同的模型来开展工作。
>
> 每个范例都是可以执行的，并打印网络结构及产生网络图表。我建议你为自己的模型做这个事情，以明确你所定义的是什么样的网络结构。
>
> 我希望这些范例能够在将来使用函数式API定义自己的模型时为您提供模板。



### 3.3标准网络模型

在开始使用函数式API时，最好先看一些标准的神经网络模型是如何定义的。

在本节中，我们将着眼于定义一个**简单的多层感知器(MLP)，卷积神经网络(CNN)和递归神经网络(RNN)**。

这些范例将为以后了解更复杂的网络构建提供基础。

#### 3.3.1多层感知器(Multilayer Perceptron)

让我们来定义了一个多类别分类的多层感知器(MLP)模型。

该模型有784个输入，3个隐藏层，512，216和128个隐藏神经元，输出层有10个输出。

在每个隐藏层中使用relu激活函数，并且在输出层中使用softmax激活函数进行多类别分类。

```python
#多层感知器(MLP)模型
from  keras.models  import  Model 
from  keras.layers  import  Input ,  Dense 
from  keras.utils  import  plot_model

mnist_input  =  Input ( shape = ( 784 ,),  name = 'input' ) 
hidden1  =  Dense ( 512 ,  activation = 'relu' ,  name = 'hidden1' )( mnist_input ) 
hidden2  =  Dense ( 216 ,  activation = 'relu' ,  name = 'hidden2' )( hidden1 ) 
hidden3  =  Dense ( 128 , activation = 'relu' ,  name = 'hidden3' )( hidden2 ) 
output  =  Dense ( 10 ,  activation = 'softmax' ,  name = 'output' )( hidden3 )

model  =  Model ( inputs = mnist_input ,  outputs = output )

#打印网络结构
model . summary ()

#产生网络拓扑图
plot_model ( model ,  to_file = 'multilayer_perceptron_graph.png' )

#秀出网络拓扑图
Image ( 'multilayer_perceptron_graph.png' )
```

------

Layer (type) Output Shape Param #
================================================== ===============
input (InputLayer) (None, 784) 0

------

hidden1 (Dense) (None, 512) 401920

------

hidden2 (Dense) (None, 216) 110808

------

hidden3 (Dense) (None, 128) 27776

------

output (Dense) (None, 10) 1290
================================================== ===============
Total params: 541,794
Trainable params: 541,794
Non-trainable params: 0

------

![如何使用Keras函数式API进行深度学习](https://img.geek-docs.com/keras/keras-examples/keras-1.1-0.png)

#### 3.3.2卷积神经网络(CNN)

我们将定义一个用于图像分类的卷积神经网络。

该模型接收灰阶的28×28图像作为输入，然后有一个作为特征提取器的两个卷积和池化层的序列，然后是一个完全连接层来解释特征，并且具有用于10类预测的softmax激活的输出层。

```python
#卷积神经网络(CNN) 
from  keras.models  import  Model 
from  keras.layers  import  Input ,  Dense 
from  keras.layers.convolutional  import  Conv2D 
from  keras.layers.pooling  import  MaxPool2D 
from  keras.utils  import  plot_model

mnist_input  =  Input ( shape = ( 28 ,  28 ,  1 ),  name = 'input' )

conv1  =  Conv2D ( 128 ,  kernel_size = 4 ,  activation = 'relu' ,  name = 'conv1' )( mnist_input ) 
pool1  =  MaxPool2D ( pool_size = ( 2 ,  2 ),  name = 'pool1' )( conv1 )

conv2  =  Conv2D ( 64 ,  kernel_size = 4 ,  activation = 'relu' ,  name = 'conv2' )( pool1 ) 
pool2  =  MaxPool2D ( pool_size = ( 2 ,  2 ),  name = 'pool2' )( conv2 )

hidden1  =  Dense ( 64 ,  activation = 'relu' ,  name = 'hidden1' )( pool2 ) 
output  =  Dense ( 10 ,  activation = 'softmax' ,  name = 'output' )( hidden1 ) 
model  =  Model ( inputs = mnist_input ,  outputs = output )

#打印网络结构
model . summary ()

#产生网络拓扑图
plot_model ( model ,  to_file = 'convolutional_neural_network.png' )

#秀出网络拓扑图
Image ( 'convolutional_neural_network.png' )
```

------

Layer (type) Output Shape Param #
================================================== ===============
input (InputLayer) (None, 28, 28, 1) 0

------

conv1 (Conv2D) (None, 25, 25, 128) 2176

------

pool1 (MaxPooling2D) (None, 12, 12, 128) 0

------

conv2 (Conv2D) (None, 9, 9, 64) 131136

------

pool2 (MaxPooling2D) (None, 4, 4, 64) 0

------

hidden1 (Dense) (None, 4, 4, 64) 4160

------

output (Dense) (None, 4, 4, 10) 650
================================================== ===============
Total params: 138,122
Trainable params: 138,122
Non-trainable params: 0

------

![如何使用Keras函数式API进行深度学习](https://img.geek-docs.com/keras/keras-examples/keras-1.1-1.png)

#### 3.3.3递归神经网络(RNN)

我们将定义一个长期短期记忆(LSTM)递归神经网络用于图像分类。

该模型预期一个特征的784个时间步骤作为输入。该模型具有单个LSTM隐藏层以从序列中提取特征， 接着是完全连接的层来解释LSTM输出，接着是用于进行10类别预测的输出层。

```python
#递归神经网络(RNN) 
from  keras.models  import  Model 
from  keras.layers  import  Input ,  Dense 
from  keras.layers.recurrent  import  LSTM 
from  keras.utils  import  plot_model

mnist_input  =  Input ( shape = ( 784 ,  1 ),  name = 'input' )  #把每一个像素想成是一序列有前后关系的time_steps 
lstm1  =  LSTM ( 128 ,  name = 'lstm1' )( mnist_input ) 
hidden1  =  Dense ( 128 ,  activation = 'relu' ,  name = 'hidden1' )( lstm1 ) 
output  =  Dense ( 10 , activation = 'softmax' ,  name = 'output' )( hidden1 ) 
model  =  Model ( inputs = mnist_input ,  outputs = output )

#打印网络结构
model . summary ()

#产生网络拓扑图
plot_model ( model ,  to_file = 'recurrent_neural_network.png' )

#秀出网络拓扑图
Image ( 'recurrent_neural_network.png' )
```

------

Layer (type) Output Shape Param #
================================================== ===============
input (InputLayer) (None, 784, 1) 0

------

lstm1 (LSTM) (None, 128) 66560

------

hidden1 (Dense) (None, 128) 16512

------

output (Dense) (None, 10) 1290
================================================== ===============
Total params: 84,362
Trainable params: 84,362
Non-trainable params: 0

------

![如何使用Keras函数式API进行深度学习](https://img.geek-docs.com/keras/keras-examples/keras-1.1-2.png)

### 3.4共享层模型

多个神经层可以共享一个神经层的输出来当成输入。

例如，一个输入可能可以有多个不同的特征提取层，或者多个神经层用于解释特征提取层的输出。

我们来看这两个例子。

#### 3.4.1共享输入层(Shared Input Layer)

我们定义具有不同大小的内核的多个卷积层来解释图像输入。

该模型使用28×28像素的灰阶图像。有两个CNN特征提取子模型共享这个输入;第一个具有4的内核大小和第二个8的内核大小。这些特征提取子模型的输出被平坦化(flatten)为向量(vector)，并且被串连成一个长向量, 然后被传递到完全连接的层以用于在最终输出层之前进行10类别预测。

```python
#共享输入层
from  keras.models  import  Model 
from  keras.layers  import  Input ,  Dense ,  Flatten 
from  keras.layers.convolutional  import  Conv2D 
from  keras.layers.pooling  import  MaxPool2D 
from  keras.layers.merge  import  concatenate 
from  keras.utils  import  plot_model

#输入层
mnist_input  =  Input ( shape = ( 28 ,  28 ,  1 ),  name = 'input' )

#第一个特征提取层
conv1  =  Conv2D ( 32 ,  kernel_size = 4 ,  activation = 'relu' ,  name = 'conv1' )( mnist_input )  # <--看这里
pool1  =  MaxPool2D ( pool_size = ( 2 ,  2 ),  name = 'pool1' )( conv1 ) 
flat1  =  Flatten ()( pool1 )

#第二个特征提取层
conv2  =  Conv2D ( 16 ,  kernel_size = 8 ,  activation = 'relu' ,  name = 'conv2' )( mnist_input )  # <--看这里
pool2  =  MaxPool2D ( pool_size = ( 2 ,  2 ),  name = 'pool2' )( conv2 ) 
flat2  =  Flatten ()( pool2 )

#把两个特征提取层的结果并起来
merge  =  concatenate ([ flat1 ,  flat2 ])

#进行全连结层
hidden1  =  Dense ( 64 ,  activation = 'relu' ,  name = 'hidden1' )( merge )

#输出层
output  =  Dense ( 10 ,  activation = 'softmax' ,  name = 'output' )( hidden1 )

#以Model来组合整个网络
model  =  Model ( inputs = mnist_input ,  outputs = output )

#打印网络结构
model . summary ()

# plot graph 
plot_model ( model ,  to_file = 'shared_input_layer.png' )

#秀出网络拓扑图
Image ( 'shared_input_layer.png' )
```

------

Layer (type) Output Shape Param # Connected to
================================================== ================================================
input (InputLayer) (None, 28, 28, 1) 0

------

conv1 (Conv2D) (None, 25, 25, 32) 544 input[0][0]

------

conv2 (Conv2D) (None, 21, 21, 16) 1040 input[0][0]

------

pool1 (MaxPooling2D) (None, 12, 12, 32) 0 conv1[0][0]

------

pool2 (MaxPooling2D) (None, 10, 10, 16) 0 conv2[0][0]

------

flatten_9 (Flatten) (None, 4608) 0 pool1[0][0]

------

flatten_10 (Flatten) (None, 1600) 0 pool2[0][0]

------

concatenate_5 (Concatenate) (None, 6208) 0 flatten_9[0][0]
flatten_10[0][0]

------

hidden1 (Dense) (None, 64) 397376 concatenate_5[0][0]

------

output (Dense) (None, 10) 650 hidden1[0][0]
================================================== ================================================
Total params: 399,610
Trainable params: 399,610
Non-trainable params: 0

------

![如何使用Keras函数式API进行深度学习](https://img.geek-docs.com/keras/keras-examples/keras-1.1-3.png)

#### 3.4.2共享特征提取层(Shared Feature Extraction Layer)

我们将使用两个并行子模型来解释用于序列分类的LSTM特征提取器的输出。

该模型的输入是1个特征的784个时间步长。具有10个存储单元的LSTM层解释这个序列。第一种解释模型是浅层单连通层， 第二层是深层3层模型。两个解释模型的输出连接成一个长向量，传递给用于进行10类别分类预测的输出层。

```python
from  keras.models  import  Model 
from  keras.layers  import  Input ,  Dense 
from  keras.layers.recurrent  import  LSTM 
from  keras.layers.merge  import  concatenate 
from  keras.utils  import  plot_model 
#输入层
mnist_input  =  Input ( shape = ( 784 ,  1 ) ,  name = 'input' )  #把每一个像素想成是一序列有前后关系的time_steps

#特征提取层
extract1  =  LSTM ( 128 ,  name = 'lstm1' )( mnist_input )

#第一个解释层
interp1  =  Dense ( 10 ,  activation = 'relu' ,  name = 'interp1' )( extract1 )  # <--看这里

#第二个解释层
interp21  =  Dense ( 64 ,  activation = 'relu' ,  name = 'interp21' )( extract1 )  # <--看这里
interp22  =  Dense ( 32 ,  activation = 'relu' ,  name = 'interp22' )( interp21 ) 
interp23  =  Dense ( 16 ,  activation = 'relu' ,  name = 'interp23' )( interp22 )

#把两个特征提取层的结果并起来
merge  =  concatenate ([ interp1 ,  interp23 ],  name = 'merge' )

#输出层
output  =  Dense ( 10 ,  activation = 'softmax' ,  name = 'output' )( merge )

#以Model来组合整个网络
model  =  Model ( inputs = mnist_input ,  outputs = output )

#打印网络结构
model . summary ()

# plot graph 
plot_model ( model ,  to_file = 'shared_feature_extractor.png' )

#秀出网络拓扑图
Image ( 'shared_feature_extractor.png' )
```

------

Layer (type) Output Shape Param # Connected to
================================================== ================================================
input (InputLayer) (None, 784, 1) 0

------

lstm1 (LSTM) (None, 128) 66560 input[0][0]

------

interp21 (Dense) (None, 64) 8256 lstm1[0][0]

------

interp22 (Dense) (None, 32) 2080 interp21[0][0]

------

interp1 (Dense) (None, 10) 1290 lstm1[0][0]

------

interp23 (Dense) (None, 16) 528 interp22[0][0]

------

merge (Concatenate) (None, 26) 0 interp1[0][0]
interp23[0][0]

------

output (Dense) (None, 10) 270 merge[0][0]
================================================== ================================================
Total params: 78,984
Trainable params: 78,984
Non-trainable params: 0

------

![如何使用Keras函数式API进行深度学习](https://img.geek-docs.com/keras/keras-examples/keras-1.1-4.png)

### 3.5多种输入和输出模型

函数式API也可用于开发具有多个输入或多个输出的模型的更复杂的模型。

#### 3.5.1多输入模型

我们将开发一个图像分类模型，将图像的两个版本作为输入，每个图像的大小不同。特别是一个灰阶的64×64版本和一个32×32的彩色版本。分离的特征提取CNN模型对每个模型进行操作，然后将两个模型的结果连接起来进行解释和最终预测。

请注意，在创建Model（）实例时，我们将两个输入图层定义为一个数组(array)。

```python
#多输入模型
from  keras.models  import  Model 
from  keras.layers  import  Input ,  Dense ,  Flatten 
from  keras.layers.convolutional  import  Conv2D 
from  keras.layers.pooling  import  MaxPool2D 
from  keras.layers.merge  import  concatenate 
from  keras.utils  import  plot_model

#第一个输入层
img_gray_bigsize  =  Input ( shape = ( 64 ,  64 ,  1 ),  name = 'img_gray_bigsize' ) 
conv11  =  Conv2D ( 32 ,  kernel_size = 4 ,  activation = 'relu' ,  name = 'conv11' )( img_gray_bigsize ) 
pool11  =  MaxPool2D ( pool_size = ( 2 ,  2 ),  name ='pool11' )( conv11 ) 
conv12  =  Conv2D ( 16 ,  kernel_size = 4 ,  activation = 'relu' ,  name = 'conv12' )( pool11 ) 
pool12  =  MaxPool2D ( pool_size = ( 2 ,  2 ),  name = 'pool12' ) ( conv12 ) 
flat1  =  Flatten ()( pool12 )

#第二个输入层
img_rgb_smallsize  =  Input ( shape = ( 32 ,  32 ,  3 ),  name = 'img_rgb_smallsize' ) 
conv21  =  Conv2D ( 32 ,  kernel_size = 4 ,  activation = 'relu' ,  name = 'conv21' )( img_rgb_smallsize ) 
pool21  =  MaxPool2D ( pool_size = ( 2 ,  2 ),  name ='pool21' )( conv21 ) 
conv22  =  Conv2D ( 16 ,  kernel_size = 4 ,  activation = 'relu' ,  name = 'conv22' )( pool21 ) 
pool22  =  MaxPool2D ( pool_size = ( 2 ,  2 ),  name = 'pool22' ) ( conv22 ) 
flat2  =  Flatten ()( pool22 )

#把两个特征提取层的结果并起来
merge  =  concatenate ([ flat1 ,  flat2 ])

#用隐藏的全连结层来解释特征
hidden1  =  Dense ( 128 ,  activation = 'relu' ,  name = 'hidden1' )( merge ) 
hidden2  =  Dense ( 64 ,  activation = 'relu' ,  name = 'hidden2' )( hidden1 )

#输出层
output  =  Dense ( 10 ,  activation = 'softmax' ,  name = 'output' )( hidden2 )

#以Model来组合整个网络
model  =  Model ( inputs = [ img_gray_bigsize ,  img_rgb_smallsize ],  outputs = output )

#打印网络结构
model . summary ()

# plot graph 
plot_model ( model ,  to_file = 'multiple_inputs.png' )

#秀出网络拓扑图
Image ( 'multiple_inputs.png' )
```

------

Layer (type) Output Shape Param # Connected to
================================================== ================================================
img_gray_bigsize (InputLayer) (None, 64, 64, 1) 0

------

img_rgb_smallsize (InputLayer) (None, 32, 32, 3) 0

------

conv11 (Conv2D) (None, 61, 61, 32) 544 img_gray_bigsize[0][0]

------

conv21 (Conv2D) (None, 29, 29, 32) 1568 img_rgb_smallsize[0][0]

------

pool11 (MaxPooling2D) (None, 30, 30, 32) 0 conv11[0][0]

------

pool21 (MaxPooling2D) (None, 14, 14, 32) 0 conv21[0][0]

------

conv12 (Conv2D) (None, 27, 27, 16) 8208 pool11[0][0]

------

conv22 (Conv2D) (None, 11, 11, 16) 8208 pool21[0][0]

------

pool12 (MaxPooling2D) (None, 13, 13, 16) 0 conv12[0][0]

------

pool22 (MaxPooling2D) (None, 5, 5, 16) 0 conv22[0][0]

------

flatten_11 (Flatten) (None, 2704) 0 pool12[0][0]

------

flatten_12 (Flatten) (None, 400) 0 pool22[0][0]

------

concatenate_6 (Concatenate) (None, 3104) 0 flatten_11[0][0]
flatten_12[0][0]

------

hidden1 (Dense) (None, 128) 397440 concatenate_6[0][0]

------

hidden2 (Dense) (None, 64) 8256 hidden1[0][0]

------

output (Dense) (None, 10) 650 hidden2[0][0]
================================================== ================================================
Total params: 424,874
Trainable params: 424,874
Non-trainable params: 0

------

![如何使用Keras函数式API进行深度学习](https://img.geek-docs.com/keras/keras-examples/keras-1.1-5.png)

#### 3.5.2多输出模型

我们将开发一个模型，进行两种不同类型的预测。给定一个特征的784个时间步长的输入序列，该模型将对该序列进行分类并输出具有相同长度的新序列。

LSTM层解释输入序列并返回每个时间步的隐藏状态。第一个输出模型创建一个堆叠的LSTM，解释这些特征，并进行多类别预测。第二个输出模型使用相同的输出层对每个输入时间步进行多类别预测。

```python
#多输出模型
from  keras.models  import  Model 
from  keras.layers  import  Input ,  Dense 
from  keras.layers.recurrent  import  LSTM 
from  keras.layers.wrappers  import  TimeDistributed 
from  keras.utils  import  plot_model

#输入层
mnist_input  =  Input ( shape = ( 784 ,  1 ),  name = 'input' )  #把每一个像素想成是一序列有前后关系的time_steps

#特征撷取层
extract  =  LSTM ( 64 ,  return_sequences = True ,  name = 'extract' )( mnist_input )

#分类输出
class11  =  LSTM ( 32 ,  name = 'class11' )( extract ) 
class12  =  Dense ( 32 ,  activation = 'relu' ,  name = 'class12' )( class11 ) 
output1  =  Dense ( 10 ,  activation = 'softmax' ,  name = 'output1' )( class12 )

#序列输出
output2  =  TimeDistributed ( Dense ( 10 ,  activation = 'softmax' ),  name = 'output2' )( extract )

#以Model来组合整个网络
model  =  Model ( inputs = mnist_input ,  outputs = [ output1 ,  output2 ])

#打印网络结构
model . summary ()

# plot graph 
plot_model ( model ,  to_file = 'multiple_outputs.png' )

#秀出网络拓扑图
Image ( 'multiple_outputs.png' )
```

------

Layer (type) Output Shape Param # Connected to
================================================== ================================================
input (InputLayer) (None, 784, 1) 0

------

extract (LSTM) (None, 784, 64) 16896 input[0][0]

------

class11 (LSTM) (None, 32) 12416 extract[0][0]

------

class12 (Dense) (None, 32) 1056 class11[0][0]

------

output1 (Dense) (None, 10) 330 class12[0][0]

------

output2 (TimeDistributed) (None, 784, 10) 650 extract[0][0]
================================================== ================================================
Total params: 31,348
Trainable params: 31,348
Non-trainable params: 0

------

![如何使用Keras函数式API进行深度学习](https://img.geek-docs.com/keras/keras-examples/keras-1.1-6.png)

#### 3.5.3最佳实践

以上有一些小技巧可以帮助你充分利用函数式API定义自己的模型。

一致性的变量名称命名对输入（可见）和输出神经层（输出）使用相同的变量名，甚至可以使用隐藏层（hidden1，hidden2）。这将有助于正确地将许多的神经层连接在一起。
检查图层摘要始终打印模型摘要并查看图层输出，以确保模型如您所期望的那样连接在一起。
查看网络拓朴图像总是尽可能地创建网络拓朴图像，并审查它，以确保一切按照你的意图连接在一起。
命名图层您可以为图层指定名称,这些名称可以让你的模型图形摘要和网络拓朴图像更容易被解读。例如：Dense（1，name =’hidden1’）。
**独立子模型考虑分离出子模型的发展，并最终将子模型结合在一起。**

### 3.6总结(Conclusion)

在这篇文章中有一些个人学习到的一些有趣的重点:

使用Keras也可以很灵活地来建构复杂的深度学习网络
每一种深度学习网络拓朴基本上都可以找的到一篇论文
了解每种深度学习网络拓朴架构的原理与应用的方向是强化内力的不二法门



## 4.Keras 从零开始构建VGG网络

Keras使得创建深度学习模型变得快速而简单, 虽然如此很多时候我们只要复制许多官网的范例就可做出很多令人觉得惊奇的结果。但是当要解决的问题需要进行一些**模型的调整与优化或是需要构建出一个新论文的网络结构**的时候, 我们就可能会左支右拙的难以招架。

在本教程中，您将通过阅读VGG的原始论文从零开始使用Keras来构建在ILSVRC-2014 (ImageNet competition)竞赛中获的第一名的VGG (Visual Geometry Group, University of Oxford)网络结构。

那么，重新构建别人已经构建的东西有什么意义呢？重点是学习。通过完成这次的练习，您将：

- 了解更多关于VGG的架构
- 了解有关卷积神经网络的更多信息
- 了解如何在Keras中实施某种网络结构
- 通过阅读论文并实施其中的某些部分可以了解更多底层的原理与原始构想

![从零开始构建VGG网络来学习Keras](https://img.geek-docs.com/keras/tutorial/vgg16.png)

为什么从VGG开始？

- 它很容易实现
- 它在ILSVRC-2014（ImageNet竞赛）上取得了优异的成绩
- 它今天被广泛使用
- 它的论文简单易懂
- Keras己经实现VGG在散布的版本中，所以你可以用来参考与比较

### 4.1让我们从论文中挖宝

根据[论文](https://arxiv.org/pdf/1409.1556v6.pdf)的测试给果D (VGG16)与E (VGG19)是效果最好的,由于这两种网络构建的方法与技巧几乎相同,因此我们选手构建D (VGG16)这个网络结构类型。

> 归纳一下论文网络构建讯息:
>
> - 输入图像尺寸( input size)：224 x 224
> - 感受过泸器( receptive field)的大小是3 x 3
> - 卷积步长( stride)是1个像素
> - 填充( padding)是1（对于3 x 3的感受过泸器）
> - 池化层的大小是2×2且步长( stride)为2像素
> - 有两个完全连接层，每层4096个神经元
> - 最后一层是具有1000个神经元的softmax分类层（代表1000个ImageNet类别）
> - 激励函数是ReLU

```python
#这个Jupyter Notebook的环境
import  platform 
import  tensorflow 
import  keras 
print ( "Platform: {} " . format ( platform . platform ())) 
print ( "Tensorflow version: {} " . format ( tensorflow . __version__ )) 
print ( " Keras version: {} " . format ( keras . __version__ ))

% matplotlib inline
 import  matplotlib.pyplot  as  plt 
import  matplotlib.image  as  mpimg 
import  numpy  as  np 
from  IPython.display  import  Image
```

> Using TensorFlow backend.
> Platform: Windows-7-6.1.7601-SP1
> Tensorflow version: 1.4.0
> Keras version: 2.1.1

### 4.2创建模型(Sequential)

```python
import  keras 
from  keras.models  import  Sequential 
from  keras.layers  import  Dense ,  Activation ,  Dropout ,  Flatten 
from  keras.layers  import  Conv2D ,  MaxPool2D 
from  keras.utils  import  plot_model

#定义输入
input_shape  =  ( 224 ,  224 ,  3 )  # RGB影像224x224 (height, width, channel)

#使用'序贯模型(Sequential)来定义
model  =  Sequential ( name = 'vgg16-sequential' )

#第1个卷积区块(block1) 
model . add ( Conv2D ( 64 ,  ( 3 ,  3 ),  padding = 'same' ,  activation = 'relu' ,  input_shape = input_shape ,  name = 'block1_conv1' )) 
model . add ( Conv2D ( 64 ,  ( 3 ,  3 ),  padding = 'same' ,  activation = 'relu' , name = 'block1_conv2' )) 
model . add ( MaxPool2D (( 2 ,  2 ),  strides = ( 2 ,  2 ),  name = 'block1_pool' ))

#第2个卷积区块(block2) 
model . add ( Conv2D ( 128 ,  ( 3 ,  3 ),  padding = 'same' ,  activation = 'relu' ,  name = 'block2_conv1' )) 
model . add ( Conv2D ( 128 ,  ( 3 ,  3 ),  padding = 'same' ,  activation = 'relu' ,  name = 'block2_conv2' ))
model . add ( MaxPool2D (( 2 ,  2 ),  strides = ( 2 ,  2 ),  name = 'block2_pool' ))

#第3个卷积区块(block3) 
model . add ( Conv2D ( 256 ,  ( 3 ,  3 ),  padding = 'same' ,  activation = 'relu' ,  name = 'block3_conv1' )) 
model . add ( Conv2D ( 256 ,  ( 3 ,  3 ),  padding = 'same' ,  activation = 'relu' ,  name = 'block3_conv2' ))
model . add ( Conv2D ( 256 ,  ( 3 ,  3 ),  padding = 'same' ,  activation = 'relu' ,  name = 'block3_conv3' )) 
model . add ( MaxPool2D (( 2 ,  2 ),  strides = ( 2 ,  2 ),  name = 'block3_pool' ))

#第4个卷积区块(block4) 
model . add ( Conv2D ( 512 ,  ( 3 ,  3 ),  padding = 'same' ,  activation = 'relu' ,  name = 'block4_conv1' )) 
model . add ( Conv2D ( 512 ,  ( 3 ,  3 ),  padding = 'same' ,  activation = 'relu' ,  name = 'block4_conv2' ))
model . add ( Conv2D ( 512 ,  ( 3 ,  3 ),  padding = 'same' ,  activation = 'relu' ,  name = 'block4_conv3' )) 
model . add ( MaxPool2D (( 2 ,  2 ),  strides = ( 2 ,  2 ),  name = 'block4_pool' ))

#第5个卷积区块(block5) 
model . add ( Conv2D ( 512 ,  ( 3 ,  3 ),  padding = 'same' ,  activation = 'relu' ,  name = 'block5_conv1' )) 
model . add ( Conv2D ( 512 ,  ( 3 ,  3 ),  padding = 'same' ,  activation = 'relu' ,  name = 'block5_conv2' ))
model . add ( Conv2D ( 512 ,  ( 3 ,  3 ),  padding = 'same' ,  activation = 'relu' ,  name = 'block5_conv3' )) 
model . add ( MaxPool2D (( 2 ,  2 ),  strides = ( 2 ,  2 ),  name = 'block5_pool' ))

#前馈全连接区块
model . add ( Flatten ( name = 'flatten' )) 
model . add ( Dense ( 4096 ,  activation = 'relu' ,  name = 'fc1' )) 
model . add ( Dense ( 4096 ,  activation = 'relu' ,  name = 'fc2' )) 
model . add ( Dense ( 1000 ,  activation= 'softmax' ,  name = 'predictions' ))

#打印网络结构
model . summary ()
```

Layer (type) Output Shape Param #
================================================== ===============
block1_conv1 (Conv2D) (None, 224, 224, 64) 1792

------

block1_conv2 (Conv2D) (None, 224, 224, 64) 36928

------

block1_pool (MaxPooling2D) (None, 112, 112, 64) 0

------

block2_conv1 (Conv2D) (None, 112, 112, 128) 73856

------

block2_conv2 (Conv2D) (None, 112, 112, 128) 147584

------

block2_pool (MaxPooling2D) (None, 56, 56, 128) 0

------

block3_conv1 (Conv2D) (None, 56, 56, 256) 295168

------

block3_conv2 (Conv2D) (None, 56, 56, 256) 590080

------

block3_conv3 (Conv2D) (None, 56, 56, 256) 590080

------

block3_pool (MaxPooling2D) (None, 28, 28, 256) 0

------

block4_conv1 (Conv2D) (None, 28, 28, 512) 1180160

------

block4_conv2 (Conv2D) (None, 28, 28, 512) 2359808

------

block4_conv3 (Conv2D) (None, 28, 28, 512) 2359808

------

block4_pool (MaxPooling2D) (None, 14, 14, 512) 0

------

block5_conv1 (Conv2D) (None, 14, 14, 512) 2359808

------

block5_conv2 (Conv2D) (None, 14, 14, 512) 2359808

------

block5_conv3 (Conv2D) (None, 14, 14, 512) 2359808

------

block5_pool (MaxPooling2D) (None, 7, 7, 512) 0

------

flatten (Flatten) (None, 25088) 0

------

fc1 (Dense) (None, 4096) 102764544

------

fc2 (Dense) (None, 4096) 16781312

------

predictions (Dense) (None, 1000) 4097000
================================================== ===============
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0

**确认模型训练的参数总数**
根据论文2.3章节的讯息与我们模型的网络结构参数比对,我们构建的模型138,357,544参数的确符合论文提及的138百万的训练参数。

### 4.3创建模型(Functaional API)

使用Keras的functiona api来定义网络结构。详细的说明与参考:

- [keras.io](https://keras.io/models/model/)
- [如何使用Keras函数式API进行深度学习](https://geek-docs.com/keras/keras-tutorial/keras-func-deeping.html)

```python
import  keras 
from  keras.models  import  Model 
from  keras.layers  import  Input ,  Dense ,  Activation ,  Dropout ,  Flatten 
from  keras.layers  import  Conv2D ,  MaxPool2D

#定义输入
input_shape  =  ( 224 ,  224 ,  3 )  # RGB影像224x224 (height, width, channel)

#输入层
img_input  =  Input ( shape = input_shape ,  name = 'img_input' )

#第1个卷积区块(block1) 
x  =  Conv2D ( 64 ,  ( 3 ,  3 ),  padding = 'same' ,  activation = 'relu' ,  name = 'block1_conv1' )( img_input ) 
x  =  Conv2D ( 64 ,  ( 3 ,  3 ),  padding = 'same' ,  activation = 'relu' ,  name = 'block1_conv2' )( x ) 
x =  MaxPool2D (( 2 ,  2 ),  strides = ( 2 ,  2 ),  name = 'block1_pool' )( x )

#第2个卷积区块(block2) 
x  =  Conv2D ( 128 ,  ( 3 ,  3 ),  padding = 'same' ,  activation = 'relu' ,  name = 'block2_conv1' )( x ) 
x  =  Conv2D ( 128 ,  ( 3 ,  3 ),  padding = 'same' ,  activation = 'relu' ,  name = 'block2_conv2' )( x ) 
x  = MaxPool2D (( 2 ,  2 ),  strides = ( 2 ,  2 ),  name = 'block2_pool' )( x )

#第3个卷积区块(block3) 
x  =  Conv2D ( 256 ,  ( 3 ,  3 ),  padding = 'same' ,  activation = 'relu' ,  name = 'block3_conv1' )( x ) 
x  =  Conv2D ( 256 ,  ( 3 ,  3 ),  padding = 'same' ,  activation = 'relu' ,  name = 'block3_conv2' )( x ) 
x  = Conv2D ( 256 ,  ( 3 ,  3 ),  padding = 'same' ,  activation = 'relu' ,  name = 'block3_conv3' )( x ) 
x  =  MaxPool2D (( 2 ,  2 ),  strides = ( 2 ,  2 ),  name = 'block3_pool' )( x )

#第4个卷积区块(block4) 
x  =  Conv2D ( 512 ,  ( 3 ,  3 ),  padding = 'same' ,  activation = 'relu' ,  name = 'block4_conv1' )( x ) 
x  =  Conv2D ( 512 ,  ( 3 ,  3 ),  padding = 'same' ,  activation = 'relu' ,  name = 'block4_conv2' )( x ) 
x  = Conv2D ( 512 ,  ( 3 ,  3 ),  padding = 'same' ,  activation = 'relu' ,  name = 'block4_conv3' )( x ) 
x  =  MaxPool2D (( 2 ,  2 ),  strides = ( 2 ,  2 ),  name = 'block4_pool' )( x )

#第5个卷积区块(block5) 
x  =  Conv2D ( 512 ,  ( 3 ,  3 ),  padding = 'same' ,  activation = 'relu' ,  name = 'block5_conv1' )( x ) 
x  =  Conv2D ( 512 ,  ( 3 ,  3 ),  padding = 'same' ,  activation = 'relu' ,  name = 'block5_conv2' )( x ) 
x  = Conv2D ( 512 ,  ( 3 ,  3 ),  padding = 'same' ,  activation = 'relu' ,  name = 'block5_conv3' )( x ) 
x  =  MaxPool2D (( 2 ,  2 ),  strides = ( 2 ,  2 ),  name = 'block5_pool' )( x )

#前馈全连接区块
x  =  Flatten ( name = 'flatten' )( x ) 
x  =  Dense ( 4096 ,  activation = 'relu' ,  name = 'fc1' )( x ) 
x  =  Dense ( 4096 ,  activation = ' relu' ,  name = 'fc2' )( x ) 
x  =  Dense ( 1000 ,  activation = 'softmax' , name = 'predictions' )( x )

#产生模型
model2  =  Model ( inputs = img_input ,  outputs = x ,  name = 'vgg16-funcapi' )

#打印网络结构
model2 . summary ()
```

------

Layer (type) Output Shape Param #
================================================== ===============
img_input (InputLayer) (None, 224, 224, 3) 0

------

block1_conv1 (Conv2D) (None, 224, 224, 64) 1792

------

block1_conv2 (Conv2D) (None, 224, 224, 64) 36928

------

block1_pool (MaxPooling2D) (None, 112, 112, 64) 0

------

block2_conv1 (Conv2D) (None, 112, 112, 128) 73856

------

block2_conv2 (Conv2D) (None, 112, 112, 128) 147584

------

block2_pool (MaxPooling2D) (None, 56, 56, 128) 0

------

block3_conv1 (Conv2D) (None, 56, 56, 256) 295168

------

block3_conv2 (Conv2D) (None, 56, 56, 256) 590080

------

block3_conv3 (Conv2D) (None, 56, 56, 256) 590080

------

block3_pool (MaxPooling2D) (None, 28, 28, 256) 0

------

block4_conv1 (Conv2D) (None, 28, 28, 512) 1180160

------

block4_conv2 (Conv2D) (None, 28, 28, 512) 2359808

------

block4_conv3 (Conv2D) (None, 28, 28, 512) 2359808

------

block4_pool (MaxPooling2D) (None, 14, 14, 512) 0

------

block5_conv1 (Conv2D) (None, 14, 14, 512) 2359808

------

block5_conv2 (Conv2D) (None, 14, 14, 512) 2359808

------

block5_conv3 (Conv2D) (None, 14, 14, 512) 2359808

------

block5_pool (MaxPooling2D) (None, 7, 7, 512) 0

------

flatten (Flatten) (None, 25088) 0

------

fc1 (Dense) (None, 4096) 102764544

------

fc2 (Dense) (None, 4096) 16781312

------

predictions (Dense) (None, 1000) 4097000
================================================== ===============
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0

------

### 4.4模型训练

要用ImageNet的资料来训练VGG16的模型则不是一件容易的事喔。

VGG论文指出：

On a system equipped with four NVIDIA Titan Black GPUs, training a single net took 2–3 weeks depending on the architecture.

也就是说就算你有四张NVIDIA的Titan网卡用Imagenet的影像集来训练VGG16模型, 可能也得花个2-3星期。即使买的起这样的硬体，你也得花蛮多的时间来训练这个模型。

**幸运的是Keras不仅己经在它的模组中包括了VGG16与VGG19的模型定义以外, 同时也帮大家预训练好了VGG16与VGG19的模型权重。**

### 4.5总结(Conclusion)

在这篇文章中有一些个人学习到的一些有趣的重点:

- 在Keras中要建构一个网络不难, 但了解这个网络架构的原理则需要多一点耐心
- VGG16构建简单效能高,真是神奇!
- VGG16在卷积层的设计是愈后面feature map的size愈小, 而过滤器(receptive field/fiter/kernel)则愈多



## 5.Keras 使用预训练的模型来分类照片中的物体

卷积神经网络现在能够在一些电脑视觉任务上胜过人类的肉眼，例如图像分类。

也就是说，定义一个物体的照片，我们可以让电脑来回答这个照片是1000个特定种类物体中的哪一类这样的问题。

> 在本教程中，我们将使用几个Keras己经内建好的预训练模型来进行图像分类，其中包括了：
>
> - VGG16
> - VGG19
> - ResNet50
> - 盗梦空间V3
> - 盗版ResNetV2
> - Xception
> - 移动网

```python
＃这个Jupyter笔记本的环境
进口 平台，
进口 tensorflow 
进口 keras 
打印（“平台：{} ” 。 格式（平台。平台（）））打印（“Tensorflow版本：{} ” 。 格式（tensorflow 。__version__ ））打印（“ keras版本：{} “ 。 格式（keras 。__version__ ））


％matplotlib直列
 进口 matplotlib.pyplot  作为 PLT 
进口 matplotlib.image  作为 mpimg 
进口 numpy的 作为 NP 
从 IPython.display  进口 图片
```

> 使用TensorFlow后端
> 平台：Windows-7-6.1.7601-SP1
> Tensorflow版本：1.4.0
> Keras版本：2.1.1



### 5.1开发一个简单的照片分类器

#### 5.1.1VGG16

##### 5.1.1.1获取样本图像

首先，我们需要一个我们可以进行分类的图像。

您可以在这里从Google随意检索一些动物照片并下载到这个jupyter notebook所处的目录。典型的说：我下载了：https://www.elephantvoices.org/images/slider/evimg16tf.jpg
![Keras 使用预训练的模型来分类照片中的物体](https://img.geek-docs.com/keras/tutorial/evimg16tf.jpg)

##### 5.1.1.2加载VGG模型

加载在Keras己经预训练好的VGG-16的权重模型档。

```python
从 keras.applications.vgg16  导入 VGG16

＃载入权重
model_vgg16  =  VGG16 （）
```

##### 5.1.1.3加载和准备图像

接下来，我们可以将图像加载进来，并转换成预训练网络所要求的张量规格。

Keras提供了一些工具来帮助完成这一步骤。

首先，我们可以使用load_img（）函数加载图像，对其大小调整为224×224所需的大小。

```python
从 keras.preprocessing.image  导入 load_img

＃载入图像档
img_file  =  'evimg16tf.jpg' 
图像 =  load_img （img_file ， target_size = （224 ， 224 ））  ＃因为VGG16的模型的输入是224x224
```

接下来，我们可以将其转换为NumPy复制，杀死我们可以在Keras中使用它。我们可以使用这个img_to_array（）函数。

```python
从 keras.preprocessing.image  导入 img_to_array

＃将图像资料转为numpy的阵列
图像 =  img_to_array （图像） ＃RGB

打印（“image.shape：” ， 图像。形状）
```

image.shape：（224，224，3）
VGG16网络期望单色阶（灰色）或多色阶图像（rgb）来作为输入；这意味着输入数组需要是转换成四个维度：

（图像尺寸，图像高，图像宽，图像色阶数）->（batch_size，img_height，img_width，img_channels）

我们只有一个样本（一个图像）。我们可以通过调用reshape（）来重新调整数组的形状，并添加额外的尺寸。

```python
＃调整张量的尺寸
image  =  image 。重塑（（1 ， 图像。形状[ 0 ]， 图像。形状[ 1 ]， 图象。形状[ 2 ]））

打印（“image.shape：” ， 图像。形状）
```

image.shape：（1，224，224，3）
接下来，我们需要按照VGG在训练ImageNet数据一样的方法来对图像进行前处理。具体而言，从论文里提出：

我们唯一要做的预处理就是从每个像素中减去在训练集上计算出的RGB平均值。

2014年，《用于深度图像识别的超深度卷积网络》。

Keras提供了一个称为preprocess_input（）的函数来为VGG网络准备新的图像输入。

```python
从 keras.applications.vgg16  导入 preprocess_input

＃准备VGG模型的图像
image  =  preprocess_input （image ）
```

##### 5.1.1.4做一个预测

我们可以调用模型中的predict（）函数来预测图像属于1000个已知对像类型的机率。

```python
＃预测所有产出类别的机率

y_pred  =  model_vgg16 。预测（图片）
```

##### 5.1.1.5解释预测

Keras提供了一个函数来解释所谓的decode_predictions（）的概率。

它可以返回一个类别的列表和每一个类别机率，为了简单起见，我们只会秀出第一个机率最高的种类。

```python
从 keras.applications.vgg16  导入 encode_predictions

＃将机率转换为类别标签
label  =  encode_predictions （y_pred ）

＃检索最可能的结果，例如最高的概率
label  =  标签[ 0 ] [ 0 ]

＃打印预测结果
print （' ％s （％.2f %% ）'  ％ （标签[ 1 ]， 标签[ 2 ] * 100 ））
```

African_elephant（30.40％）

#### 5.1.2VGG19

```python
从 keras.applications.vgg19  进口 VGG19 
从 keras.preprocessing  导入 图像
从 keras.applications.vgg19  进口 preprocess_input 
从 keras.applications.vgg19  进口 decode_predictions

＃载入权重
model_vgg19  =  VGG19 （权重= 'imagenet' ）＃
载入图像档
img_file  =  'evimg16tf.jpg' 
图像 =  load_img （img_file ， target_size = （224 ， 224 ））  ＃因为VGG19的模型的输入是224x224

＃将图像资料转为numpy的阵列
图像 =  img_to_array （图像） ＃RGB 
打印（“image.shape：” ， 图像。形状）

＃调整张量的尺寸
image  =  image 。重塑（（1 ， 图像。形状[ 0 ]， 图像。形状[ 1 ]， 图象。形状[ 2 ]））
的打印（“image.shape：” ， 图像。形状）

＃准备模型所需要的图像前处理
image  =  preprocess_input （image ）

＃预测所有
产出 类别的机率y_pred =  model_vgg19 。预测（图片）

＃将机率转换为类别标签
label  =  encode_predictions （y_pred ）

＃检索最可能的结果，例如最高的概率
label  =  标签[ 0 ] [ 0 ]

＃打印预测结果
print （' ％s （％.2f %% ）'  ％ （标签[ 1 ]， 标签[ 2 ] * 100 ）
```

image.shape：（224，224，3）
image.shape：（1，224，224，3）
图斯克（53.61％）

#### 5.1.3ResNet50

```python
从 keras.applications.resnet50  进口 ResNet50 
从 keras.preprocessing  导入 图像
从 keras.applications.resnet50  进口 preprocess_input 
从 keras.applications.resnet50  进口 decode_predictions

＃载入权重
model_resnet50  =  ResNet50 （权重= 'imagenet' ）

＃
加载 图像档img_file =  'evimg16tf.jpg'

＃因为RESNET的模型的输入是224x224 
图像 =  load_img （img_file ， target_size = （224 ， 224 ）） 

＃将图像资料转为numpy的阵列
图像 =  img_to_array （图像） ＃RGB 
打印（“image.shape：” ， 图像。形状）

＃调整张量的尺寸
image  =  image 。重塑（（1 ， 图像。形状[ 0 ]， 图像。形状[ 1 ]， 图象。形状[ 2 ]））
的打印（“image.shape：” ， 图像。形状）

＃准备模型所需要的图像前处理
image  =  preprocess_input （image ）

＃预测所有
产出 类别的机率y_pred =  model_resnet50 。预测（图片）

＃将机率转换为类别标签
label  =  encode_predictions （y_pred ）

＃检索最可能的结果，例如最高的概率
label  =  标签[ 0 ] [ 0 ]

＃打印预测结果
print （' ％s （％.2f %% ）'  ％ （标签[ 1 ]， 标签[ 2 ] * 100 ））
```

image.shape：（224，224，3）
image.shape：（1，224，224，3）
African_elephant（47.35％）

#### 5.1.4InceptionV3

```python
从 keras.applications.inception_v3  进口 InceptionV3 
从 keras.preprocessing  导入 图像
从 keras.preprocessing.image  进口 load_img 
从 keras.applications.inception_v3  进口 preprocess_input 
从 keras.applications.inception_v3  进口 decode_predictions

＃
加载 权重model_inception_v3 =  InceptionV3 （权重= 'imagenet' ）

＃载入图像档
img_file  =  'image.jpg的' 
＃InceptionV3的模型的输入是299x299 
IMG  =  load_img （img_file ， target_size = （299 ， 299 ）） 

＃将图像资料转为numpy array 
image  =  image 。img_to_array （IMG ） ＃RGB 
打印（“image.shape：” ， 图像。形状）

＃调整张量的尺寸
image  =  image 。重塑（（1 ， 图像。形状[ 0 ]， 图像。形状[ 1 ]， 图象。形状[ 2 ]））
的打印（“image.shape：” ， 图像。形状）

＃准备模型所需要的图像前处理
image  =  preprocess_input （image ）

＃预测所有
产出 类别的机率y_pred =  model_inception_v3 。预测（图片）

＃将机率转换为类别标签
label  =  encode_predictions （y_pred ）

＃检索最可能的结果，例如最高的概率
label  =  标签[ 0 ] [ 0 ]

＃打印预测结果
print （' ％s （％.2f %% ）'  ％ （标签[ 1 ]， 标签[ 2 ] * 100 ））
```

image.shape：（299，299，3）
image.shape：（1、299、299、3）
大熊猫（94.42％）

#### 5.1.5InceptionResNetV2

```python
从 keras.applications.inception_resnet_v2  进口 InceptionResNetV2 
从 keras.preprocessing  导入 图像
从 keras.applications.inception_resnet_v2  进口 preprocess_input 
从 keras.applications.inception_resnet_v2  进口 decode_predictions

＃
加载 权重model_inception_resnet_v2 =  InceptionResNetV2 （权重= 'imagenet' ）

＃
加载 图像档img_file =  'evimg16tf.jpg'

＃InceptionResNetV2的模型的输入是299x299 
图像 =  load_img （img_file ， target_size = （299 ， 299 ）） 

＃将图像资料转为numpy的阵列
图像 =  img_to_array （图像） ＃RGB 
打印（“image.shape：” ， 图像。形状）

＃调整张量的尺寸
image  =  image 。重塑（（1 ， 图像。形状[ 0 ]， 图像。形状[ 1 ]， 图象。形状[ 2 ]））
的打印（“image.shape：” ， 图像。形状）

＃准备模型所需要的图像前处理
image  =  preprocess_input （image ）

＃预测所有目标类别的机率
y_pred  =  model_inception_resnet_v2 。预测（图片）

＃将机率转换为类别标签
label  =  encode_predictions （y_pred ）

＃检索最可能的结果，例如最高的概率
label  =  标签[ 0 ] [ 0 ]

＃打印预测结果
print （' ％s （％.2f %% ）'  ％ （标签[ 1 ]， 标签[ 2 ] * 100 ））
```

image.shape：（299，299，3）
image.shape：（1、299、299、3）
African_elephant（62.94％）

#### 5.1.6MobileNet

```python
从 keras.applications.mobilenet  进口 MobileNet 
从 keras.preprocessing  导入 图像
从 keras.applications.mobilenet  进口 preprocess_input 
从 keras.applications.mobilenet  进口 decode_predictions

＃
加载 权重model_mobilenet =  MobileNet （权重= 'imagenet' ）

＃
加载 图像档img_file =  'evimg16tf.jpg'

＃MobileNet的模型的输入是224x224 
图像 =  load_img （img_file ， target_size = （224 ， 224 ）） 

＃将图像资料转为numpy的阵列
图像 =  img_to_array （图像） ＃RGB 
打印（“image.shape：” ， 图像。形状）

＃调整张量的尺寸
image  =  image 。重塑（（1 ， 图像。形状[ 0 ]， 图像。形状[ 1 ]， 图象。形状[ 2 ]））
的打印（“image.shape：” ， 图像。形状）

＃准备模型所需要的图像前处理
image  =  preprocess_input （image ）

＃预测所有目标类别的机率
y_pred  =  model_mobilenet 。预测（图片）

＃将机率转换为类别标签
label  =  encode_predictions （y_pred ）

＃检索最可能的结果，例如最高的概率
label  =  标签[ 0 ] [ 0 ]

＃打印预测结果
print （' ％s （％.2f %% ）'  ％ （标签[ 1 ]， 标签[ 2 ] * 100 ））
```

image.shape：（224，224，3）
image.shape：（1，224，224，3）
非洲象（90.81％）

### 5.2总结（结论）

在这篇文章中有一些个人学习到的一些有趣的重点：

- 在Keras中己经预建了许多高级的图像识别的网络及预训练的权重
- 需要了解每个高级图像识别的网络的结构与输入的张量
- 了解不同高级图像识别的网络的训练变数量与预训练的权重可以有效帮助图像识别类型的任务



## 6.Keras 如何训练小数据集

一个想要非常的数据来训练分类的模型是一种在实务上常见的情况，如果您过去曾经进行过影像视觉处理的相关专案，这样的情况势必会很少出现。

训练用样本很“少”可以示范从小动物到几万个图像（视不同的应用与场景）。作为一个示范的案例，我们将集中在将图像分类为“狗”或“猫”，数据集中包含4000张猫和狗照片（2000只猫，2000只狗）。我们将使用2000张图片进行训练，1000张用于验证，最后1000张用于测试。

我们将回顾一个解决这一问题的基本策略：从零开始，我们提供少量数据来训练一个新的模型。我们将首先在我们的 2000 个训练样本上简单地训练一个小型大众网络（convnets）来做为未来优化调整的基准，在这过程中没有任何正规化（正规化）的修补或配置。

在这个阶段，我们的主要问题将是过拟合（过拟合）。然后，我们将介绍数据扩充（一种数据扩充），这是通过利用数据扩充（数据扩充），我们将改进计算机视觉演算智能手机，并提升准确率达到82%。

在另一个文章中，我们将探索另外两种将深度学习到小数据集的基本技术：使用预训练的网络模型来进行特征提取（这使我们达到了 90% 至 93% 的准确应用率），调整一个采集训练的网络模型（这队我们达到95%的最终准确率）。总而言之，以下三个策略：

- 从头开始训练一个小型模型
- 使用开采训练的模型进行特征提取
- 微调训练的模型

将构成您未来的工具箱，用于解决计算机视觉运算应用到小数据集的问题上。

```python
#这个Jupyter Notebook的环境
import  platform 
import  tensorflow 
import  keras 
print ( "Platform: {} " . format ( platform . platform ())) 
print ( "Tensorflow version: {} " . format ( tensorflow . __version__ )) 
print ( " Keras version: {} " . format ( keras . __version__ ))

% matplotlib inline
 import  matplotlib.pyplot  as  plt 
import  matplotlib.image  as  mpimg 
import  numpy  as  np 
from  IPython.display  import  Image
```

> 使用 TensorFlow 后端。
> 平台：Windows-10-10.0.15063-SP0 
> Tensorflow 版本：1.4.0 Keras
> 版本：2.0.9

### 6.1资料集说明

我们将使用的数据集（猫与狗的图片集）没有被包装Keras包装发布，所以要自己另外下载。Kaggle.com在2013年底提供了这些数据来作为电脑游戏竞赛题目。您可以从以下链接下载原始数据集：https://www.kaggle.com/c/dogs-vs-cats/data

图片是中等解析度的彩色压缩文件。他们看起来像这样：本来
![Keras 如何训练小数据集](https://img.geek-docs.com/keras/tutorial/cats_vs_dogs_samples.jpg)
数据集包含 25,000 张狗和猫的平均图像（类别 12,500 个），大小为 543MB（）。下载和解压缩后，我们将创建一个包含三个子集的新数据集：一组包含每个样本的 100 个样本的训练集，每组 500 个样本的验证集，最后一个包含每个样本的 500 个样本的测试集。

### 6.2资料准备

1. 从[Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data)点击下载下载图像资料档次培训.zip。
2. 在这个 Jupyter Notebook 的目录下产生一个新的子目录“数据”。
3. 把从kaggle下载的资料档复制到“资料”的目录里头。
4. 将train.zip解压缩

你的最后一个目录结构像这样：

xx-yyy.ipynb
数据/ 
└── train/ 
├── cat.0.jpg 
├── cat.1.jpg 
├── .. 
└── dog.12499.jpg

```python
import  os

#专案的根目录路径
ROOT_DIR  =  os . getcwd ()

#置放coco图像资料与标注资料的目录
DATA_PATH  =  os . path . join ( ROOT_DIR ,  "data" )
```

```python
import  os ,  shutil

#原始数据集的路径
original_dataset_dir  =  os . path . join ( DATA_PATH ,  "train" )

#存储小数据集的目录
base_dir  =  os . path . join ( DATA_PATH ,  "cats_and_dogs_small" ) 
if  not  os . path . exists ( base_dir ):  
    os . mkdir ( base_dir )

#我们的训练资料的目录
train_dir  =  os . path . join ( base_dir ,  'train' ) 
if  not  os . path . exists ( train_dir ):  
    os . mkdir ( train_dir )

#我们的验证资料的目录
validation_dir  =  os . path . join ( base_dir ,  'validation' ) 
if  not  os . path . exists ( validation_dir ):  
    os . mkdir ( validation_dir )

#我们的测试资料的目录
test_dir  =  os . path . join ( base_dir ,  'test' ) 
if  not  os . path . exists ( test_dir ): 
    os . mkdir ( test_dir )    

#猫的图片的训练资料目录
train_cats_dir  =  os . path . join ( train_dir ,  'cats' ) 
if  not  os . path . exists ( train_cats_dir ): 
    os . mkdir ( train_cats_dir )

#狗的图片的训练资料目录
train_dogs_dir  =  os . path . join ( train_dir ,  'dogs' ) 
if  not  os . path . exists ( train_dogs_dir ): 
    os . mkdir ( train_dogs_dir )

#猫的图片的验证资料目录
validation_cats_dir  =  os . path . join ( validation_dir ,  'cats' ) 
if  not  os . path . exists ( validation_cats_dir ): 
    os . mkdir ( validation_cats_dir )

#狗的图片的验证资料目录
validation_dogs_dir  =  os . path . join ( validation_dir ,  'dogs' ) 
if  not  os . path . exists ( validation_dogs_dir ): 
    os . mkdir ( validation_dogs_dir )

#猫的图片的测试资料目录
test_cats_dir  =  os . path . join ( test_dir ,  'cats' ) 
if  not  os . path . exists ( test_cats_dir ): 
    os . mkdir ( test_cats_dir )

#狗的图片的测试资料目录
test_dogs_dir  =  os . path . join ( test_dir ,  'dogs' ) 
if  not  os . path . exists ( test_dogs_dir ): 
    os . mkdir ( test_dogs_dir )
```

```python
#复制前1000个猫的图片到train_cats_dir 
fnames  =  [ 'cat. {} .jpg' . format ( i )  for  i  in  range ( 1000 )] 
for  fname  in  fnames : 
    src  =  os . path . join ( original_dataset_dir ,  fname ) 
    dst  =  os . path . join ( train_cats_dir ,  fname ) 
    if  not  os .path . exists ( dst ): 
        shutil . copyfile ( src ,  dst )

print ( 'Copy first 1000 cat images to train_cats_dir complete!' )

#复制下500个猫的图片到validation_cats_dir 
fnames  =  [ 'cat. {} .jpg' . format ( i )  for  i  in  range ( 1000 ,  1500 )] 
for  fname  in  fnames : 
    src  =  os . path . join ( original_dataset_dir ,  fname ) 
    dst  =  os . path . join ( validation_cats_dir ,  fname ) 
    if not  os . path . exists ( dst ): 
        shutil . copyfile ( src ,  dst )

print ( 'Copy next 500 cat images to validation_cats_dir complete!' )

#复制下500个猫的图片到test_cats_dir 
fnames  =  [ 'cat. {} .jpg' . format ( i )  for  i  in  range ( 1500 ,  2000 )] 
for  fname  in  fnames : 
    src  =  os . path . join ( original_dataset_dir ,  fname ) 
    dst  =  os . path . join ( test_cats_dir ,  fname ) 
    if  not os . path . exists ( dst ): 
        shutil . copyfile ( src ,  dst )

print ( 'Copy next 500 cat images to test_cats_dir complete!' )

#复制前1000个狗的图片到train_dogs_dir 
fnames  =  [ 'dog. {} .jpg' . format ( i )  for  i  in  range ( 1000 )] 
for  fname  in  fnames : 
    src  =  os . path . join ( original_dataset_dir ,  fname ) 
    dst  =  os . path . join ( train_dogs_dir ,  fname ) 
    if  not  os .path . exists ( dst ): 
        shutil . copyfile ( src ,  dst )

print ( 'Copy first 1000 dog images to train_dogs_dir complete!' )


#复制下500个狗的图片到validation_dogs_dir 
fnames  =  [ 'dog. {} .jpg' . format ( i )  for  i  in  range ( 1000 ,  1500 )] 
for  fname  in  fnames : 
    src  =  os . path . join ( original_dataset_dir ,  fname ) 
    dst  =  os . path . join ( validation_dogs_dir ,  fname ) 
    if not  os . path . exists ( dst ): 
        shutil . copyfile ( src ,  dst )

print ( 'Copy next 500 dog images to validation_dogs_dir complete!' )

# C复制下500个狗的图片到test_dogs_dir 
fnames  =  [ 'dog. {} .jpg' . format ( i )  for  i  in  range ( 1500 ,  2000 )] 
for  fname  in  fnames : 
    src  =  os . path . join ( original_dataset_dir ,  fname ) 
    dst  =  os . path . join ( test_dogs_dir ,  fname ) 
    if  not os . path . exists ( dst ): 
        shutil . copyfile ( src ,  dst )

print ( 'Copy next 500 dog images to test_dogs_dir complete!' )
```

> 将前 1000 张猫图像复制到 train_cats_dir 完成！
> 将接下来的 500 张猫图像复制到 validation_cats_dir 完成！
> 将接下来的 500 张猫图像复制到 test_cats_dir 完成！
> 将前 1000 张狗图像复制到 train_dogs_dir 完成！
> 将接下来的 500 个狗图像复制到 validation_dogs_dir 完成！
> 将接下来的 500 个狗图像复制到 test_dogs_dir 完成！

一个健康检查，让我们计算一下有多少张照片（训练验证/测试）：

```python
print ( 'total training cat images:' ,  len ( os . listdir ( train_cats_dir ))) 
print ( 'total training dog images:' ,  len ( os . listdir ( train_dogs_dir ))) 
print ( 'total validation cat images:' ,  len ( os . listdir ( validation_cats_dir ))) 
print ( 'total validation dog images:' ,  len ( os . listdir (validation_dogs_dir ))) 
print ( 'total test cat images:' ,  len ( os . listdir ( test_cats_dir ))) 
print ( 'total test dog images:' ,  len ( os . listdir ( test_dogs_dir )))
```

总训练猫图：1000
总训练狗图像：1000
总验证猫图像：500
总验证狗图像：500
总测试猫图像：500
总测试狗图像：500
所以我们确实有2000 个训练图像，然后是1000 个验证图像和1000个测试图像。在每个数据分割（分割）中，每个分类相同数量的样本：这是一个平衡的二元分类问题，这表明分类准确度将适当的研究。

### 6.3资料预处理（Data Preprocessing）

如现在已经知道的那样，数据应该被格式化成适当的浮点张，然后才能进入我们的神经网络。目前，我们的数据是在档案目里里的JPEG影像文件，所以进入我们的网络的前处理步骤大概是：

- 读进图像档案。
- 将JPEG内容解码为RGB像素的像素。
- 将其转换为浮点张量。
- 将像素值（0和255之间）放大到[0,1]间隔（如您，神经网络更喜欢处理输入值）。

这可能看起来令人生畏，但感谢 Keras 有一些工具程序可以自动处理这些。Keras 有一个图像处理助手工具的模块，位于 keras.preprocessing.image。其中的 ImageDataGenerator 类别，可以快速的自动将磁盘磁盘上的图像文件转换成张量(tensors)。我们将在这里使用这个工具。

```python
from  keras.preprocessing.image  import  ImageDataGenerator

#所有的图像将重新被进行归一化处理Rescaled by 1./255 
train_datagen  =  ImageDataGenerator ( rescale = 1. / 255 ) 
test_datagen  =  ImageDataGenerator ( rescale = 1. / 255 )

#直接从档案目录读取图像档资料
train_generator  =  train_datagen . flow_from_directory (  
        #这是图像资料的目录
        train_dir , 
        #所有的图像大小会被转换成150x150 
        target_size = ( 150 ,  150 ), 
        #每次产生20图像的批次资料
        batch_size = 20 , 
        #由于这是一个二元分类问题, y的lable值也会被转换成二元的标签
        class_mode = 'binary' )

#直接从档案目录读取图像档资料
validation_generator  =  test_datagen . flow_from_directory ( 
        validation_dir , 
        target_size = ( 150 ,  150 ), 
        batch_size = 20 , 
        class_mode = 'binary' )
```

找到属于 2 个类别的 2000 张图像。
找到1000张图片，属于2类。
我们来看看这些图片张量生成器(generator)的输出：它产生150×150 RGB图片（形状“（20,150,150,3）”）和二进制标签（形状“（20， ）”）”）大约的距离远张量。 20 是可视中的数字（大小）。请注意，无限制地产生这些传感器：因为它只是持续地循环的目标中存在的图像。因此，我们需要在一段时间内中断循环。

```python
for  data_batch ,  labels_batch  in  train_generator : 
    print ( 'data batch shape:' ,  data_batch . shape ) 
    print ( 'labels batch shape:' ,  labels_batch . shape ) 
    break
```

data batch shape: (20, 150, 150, 3) 
labels batch shape: (20,)
让我们将模型与使用图像张量生成器的数据进行训练。我们使用fit_generator方法。

因为数据是可以无休止地持续生成，所以图像张量产生器需要知道在一个训练循环（纪元）要从图像张量产生器中爆发大量个数据。这是steps_per_epoch参数的作用：在从生成器中跑过s_per_epochsteps之后，即在运行steps_per_epoch下降步骤之后，训练过程将转到下一个循环（epoch）。在我们的情况下，是20个样本，所以需要100次，直到我们的模型读进了2000个目标样本。

当使用fit_generator时，可以传递一个validation_data参数，这个就像fit方法一样。重要的是，参数被允许作为数据生成器制作，但它也是一个Numpy数组的元组。如果您将生成器可以传递为验证_数据，那么这个生成器你的钉周圈生成数据，因此还应该进行验证_步骤，该参数告诉过程中从验证生成器指定的验证生成器中广场人群以进行评估。

### 6.4网络模型（模型）

我们的神经网络(nets)将是一组卷积的Conv2D（具有relu激活）和MaxPooling2D层。我们从大小150×150（任意选择）输入的开始，我们最终得到了尺寸为7×7的Flatten层之前的特征图。

注意，特征图的深度在网络中逐渐增加（从328到128），而特征图的大小正在减少（从148×148到7×7）。这是你将在几乎所有的神经网络（convnets） )装中会看到的模式。

因为我们正在处理二元分类问题，所以我们用一个神经元（一个大小为1的密集层（密集））和一个sigmoid激活函数来结束网络。类或另一类的机率。

```python
from  keras  import  layers 
from  keras  import  models 
from  keras.utils  import  plot_model

model  =  models . Sequential () 
model . add ( layers . Conv2D ( 32 ,  ( 3 ,  3 ),  activation = 'relu' , 
                        input_shape = ( 150 ,  150 ,  3 ))) 
model . add ( layers . MaxPooling2D (( 2 ,  2 ))) 
model . add ( layers . Conv2D( 64 ,  ( 3 ,  3 ),  activation = 'relu' )) 
model . add ( layers . MaxPooling2D (( 2 ,  2 ))) 
model . add ( layers . Conv2D ( 128 ,  ( 3 ,  3 ),  activation = ' relu' )) 
model . add ( layers . MaxPooling2D (( 2 ,  2))) 
model . add ( layers . Conv2D ( 128 ,  ( 3 ,  3 ),  activation = 'relu' )) 
model . add ( layers . MaxPooling2D (( 2 ,  2 ))) 
model . add ( layers . Flatten () ) 
model . add ( layers . Dense ( 512 ,  activation = 'relu')) 
model . add ( layers . Dense ( 1 ,  activation = 'sigmoid' ))
```

我们来看看每一幅画面的尺寸如何随着生活的画面而改变：

```python
#打印网络结构
model . summary ()
```

层（类型）输出形状参数# 
==================================================================================================================================================================================================================================================================================================================================================================================================================================== ======== ================ 
conv2d_1 (Conv2D) (无, 148, 148, 32) 896

------

max_pooling2d_1 (MaxPooling2 (None, 74, 74, 32) 0

------

conv2d_2 (Conv2D)（无、72、72、64）18496

------

max_pooling2d_2 (MaxPooling2 (None, 36, 36, 64) 0

------

conv2d_3 (Conv2D)（无、34、34、128）73856

------

max_pooling2d_3 (MaxPooling2 (None, 17, 17, 128) 0

------

conv2d_4 (Conv2D)（无、15、15、128）147584

------

max_pooling2d_4 (MaxPooling2 (None, 7, 7, 128) 0

------

flatten_1（扁平化）（无，6272）0

------

密集_1（密集）（无，512）3211776

------

密集_2（密集）（无，1）513 
======================================== ========== ================
总参数：3,453,121 可训练参数：
3,453,121
不可训练参数：0

------

由于我们使用了一个单一的神经元（Sigmoid 的激活函数）结束了的网络，我们将使用我们的双熵（二进制交叉熵）作为损失函数。

```python
from  keras  import  optimizers

model . compile ( loss = 'binary_crossentropy' , 
              optimizer = optimizers . RMSprop ( lr = 1e-4 ), 
              metrics = [ 'acc' ])
```

### 6.5训练（训练）

```python
history  =  model . fit_generator ( 
      train_generator , 
      steps_per_epoch = 100 , 
      epochs = 30 , 
      validation_data = validation_generator , 
      validation_steps = 50 )
```

Epoch 1/30 
100/100 [==============================] – 11s 112ms/step – loss: 0.6955 – acc : 0.5295 – val_loss : 0.6809 – val_acc: 0.5340 
Epoch 2/30 
100/100 [==============================] – 8s 78ms/step – loss: 0.6572 – acc: 0.6135 – val_loss : 0.6676 – val_acc: 0.5710 
Epoch 3/30 
100/100 [====================== ========] – 8s 76ms/step – loss: 0.6214 – acc: 0.6590 – val_loss : 0.6408 – val_acc: 0.6290 
Epoch 4/30 
100/100 [============ ==================] – 8s 77ms/step – loss: 0.5882 – acc: 0.6920 – val_loss : 0.6296 – val_acc: 0.6440 
Epoch 5/30 
100/100 [== ============================] – 8s 77ms/step – loss: 0.5517 – acc: 0.7130 – val_loss : 0.7241 – val_acc: 0.5950
时代 6/30
100/100 [==============================] – 8s 76ms/step – loss: 0.5108 – acc: 0.7530 – val_loss : 0.5785 – val_acc: 0.7010 
Epoch 7/30 
100/100 [==============================] – 8s 78ms/step – loss: 0.4899 – acc: 0.7680 – val_loss : 0.5620 – val_acc: 0.7060 
Epoch 8/30 
100/100 [========================== ====] – 8s 76ms/step – loss: 0.4575 – acc: 0.7810 – val_loss : 0.5803 – val_acc: 0.6970 
Epoch 9/30 
100/100 [================ ==============] – 8s 76ms/step – loss: 0.4193 – acc: 0.8070 – val_loss : 0.5881 – val_acc: 0.7120 
Epoch 10/30 
100/100 [====== ========================] – 8s 75ms/step – loss: 0.3869 – acc: 0.8195 – val_loss : 0.5986 – val_acc: 0.7050 
Epoch 11/30
100/100 [==============================] – 8s 75ms/step – loss: 0.3620 – acc: 0.8355 – val_loss : 0.6368 – val_acc: 0.7090 
Epoch 12/30 
100/100 [==============================] – 8s 76ms/step – loss: 0.3434 – acc: 0.8480 – val_loss : 0.6214 – val_acc: 0.6970 
Epoch 13/30 
100/100 [========================== ====] – 8s 75ms/step – loss: 0.3165 – acc: 0.8670 – val_loss : 0.6897 – val_acc: 0.7010 
Epoch 14/30 
100/100 [================ ==============] – 8s 76ms/step – loss: 0.2878 – acc: 0.8755 – val_loss : 0.6249 – val_acc: 0.7100 
Epoch 15/30 
100/100 [====== ========================] – 8s 77ms/step – loss: 0.2650 – acc: 0.8975 – val_loss : 0.6438 – val_acc: 0.7060 
Epoch 16/30
100/100 [==============================] – 8s 76ms/step – loss: 0.2362 – acc: 0.9090 – val_loss : 0.7780 – val_acc: 0.6920 
Epoch 17/30 
100/100 [==============================] – 8s 76ms/step – loss: 0.2098 – acc: 0.9165 – val_loss : 0.8215 – val_acc: 0.6750 
Epoch 18/30 
100/100 [========================== ====] – 8s 76ms/step – loss: 0.1862 – acc: 0.9305 – val_loss : 0.7044 – val_acc: 0.7120 
Epoch 19/30 
100/100 [================ ==============] – 8s 75ms/step – loss: 0.1669 – acc: 0.9425 – val_loss : 0.7941 – val_acc: 0.6990 
Epoch 20/30 
100/100 [====== ========================] – 8s 75ms/step – loss: 0.1522 – acc: 0.9475 – val_loss : 0.8285 – val_acc: 0.6960 
Epoch 21/30
100/100 [==============================] – 8s 75ms/step – loss: 0.1254 – acc: 0.9575 – val_loss : 0.8199 – val_acc: 0.7070 
Epoch 22/30 
100/100 [==============================] – 8s 78ms/step – loss: 0.1117 – acc: 0.9620 – val_loss : 0.9325 – val_acc: 0.7090 
Epoch 23/30 
100/100 [========================== ====] – 8s 76ms/step – loss: 0.0907 – acc: 0.9750 – val_loss : 0.8740 – val_acc: 0.7220 
Epoch 24/30 
100/100 [================ ==============] – 8s 75ms/step – loss: 0.0806 – acc: 0.9755 – val_loss : 1.0178 – val_acc: 0.6900 
Epoch 25/30 
100/100 [====== ========================] – 8s 75ms/step – loss: 0.0602 – acc: 0.9815 – val_loss : 0.9158 – val_acc: 0.7260 
Epoch 26/30
100/100 [==============================] – 8s 76ms/step – loss: 0.0591 – acc: 0.9810 – val_loss : 1.1284 – val_acc: 0.7030 
Epoch 27/30 
100/100 [==============================] – 8s 75ms/step – loss: 0.0511 – acc: 0.9820 – val_loss : 1.1136 – val_acc: 0.7140 
Epoch 28/30 
100/100 [========================== ====] – 8s 76ms/step – loss: 0.0335 – acc: 0.9930 – val_loss : 1.4372 – val_acc: 0.6820 
Epoch 29/30 
100/100 [================ ==============] – 8s 75ms/step – loss: 0.0409 – acc: 0.9860 – val_loss : 1.2121 – val_acc: 0.6930 
Epoch 30/30 
100/100 [====== ========================] – 8s 76ms/step – loss: 0.0271 – acc: 0.9920 – val_loss : 1.3055 – val_acc: 0.7010
训练完后就把模型保存是个好习惯：

```python
model.save ( 'cats_and_dogs_small_2.h5' )
```

我们用图表来秀出在训练过程中模型对训练和验证数据的损失（损失）和准确（准确性）数据：

```python
import  matplotlib.pyplot  as  plt

acc  =  history . history [ 'acc' ] 
val_acc  =  history . history [ 'val_acc' ] 
loss  =  history . history [ 'loss' ] 
val_loss  =  history . history [ 'val_loss' ]

epochs  =  range ( len ( acc ))

plt . plot ( epochs ,  acc ,  label = 'Training acc' ) 
plt . plot ( epochs ,  val_acc ,  label = 'Validation acc' ) 
plt . title ( 'Training and validation accuracy' ) 
plt . legend ()

plt . figure ()

plt . plot ( epochs ,  loss ,  label = 'Training loss' ) 
plt . plot ( epochs ,  val_loss ,  label = 'Validation loss' ) 
plt . title ( 'Training and validation loss' ) 
plt . legend ()

plt . show ()
```

![Keras 如何训练小数据集](https://img.geek-docs.com/keras/tutorial/xiaoshuju-1.png)
![Keras 如何训练小数据集](https://img.geek-docs.com/keras/tutorial/xiaoshuju-2.png)

我们的训练强度随着时间的增长，到接近100%，我们的验证程度却停在70~72%。个（eps）之后达到循环，然后停顿，而训练损失在线性上保持直到接近0。

因为只有智慧的训练，我们可以相对的20笔数据了（L2），我们可以成为真正的重点关注点。正规化）。我们现在要推出一个新的，特定于视觉影像，并在使用电脑学习模型处理图像时几乎可以使用的技巧：数据扩充（数据扩充）。

### 6.6使用数据膨胀

凯特（过度拟合）是由于样本数量太少而导致的，导致我们无法推广到新数据的模型。

给定无限数据，我们的模型将暴露在手头的数据分发到最可能的方面：我们永远不会忘记的。采用从现有方法训练样本生成更多训练数据的，通过产生的可信的图像的多个随机变换来“增加”样本。目标是在训练的时候，我们的模型不会再出现完全相同的画面。这模型模型呈现于数据的更多方面，并更好地推广。

在Keras中，可以通过配置对我们的ImageDataGenerator实例读取的图像执行多个随机动作来完成。让我们开始一个例子：

```python
datagen  =  ImageDataGenerator ( 
      rotation_range = 40 , 
      width_shift_range = 0.2 , 
      height_shift_range = 0.2 , 
      shear_range = 0.2 , 
      zoom_range = 0.2 , 
      horizontal_flip = True , 
      fill_mode = 'nearest' )
```

只是这些还有一些可用的选项（更多选项，请参阅 Keras 文档）。我们快速看一下这些参数：

- 旋转范围是以度（0-180）为单位的值，是随机旋转图片的范围。
- width_shift和height_shift是范围（占总长度或高度的部分），用于纵向或随机转换图片。
- shear_range 用于随机剪切变换。
- zoom_range 用于随机放大图片内容。
- horizontal_flip 用于在没有水平不假的情况下（例如真实世界图片）的情况下水平地随机动画图像。
- _mode是用于显示填充新创建的像素的策略，可以在旋转或宽/高填充后。

我们来看看我们的增益后图片：

```python
import  matplotlib.pyplot  as  plt 
from  keras.preprocessing  import  image 
#取得训练资料集中猫的档案列表
fnames  =  [ os . path . join ( train_cats_dir ,  fname )  for  fname  in  os . listdir ( train_cats_dir )]

#取一个图像
img_path  =  fnames [ 3 ]

#读图像并进行大小处理
img  =  image . load_img ( img_path ,  target_size = ( 150 ,  150 ))

#转换成Numpy array并且shape (150, 150, 3) 
x  =  image . img_to_array ( img )

#重新Reshape成(1, 150, 150, 3)以便输入到模型中
x  =  x . reshape (( 1 ,)  +  x . shape )

#透过flow()方法将会随机产生新的图像
#它会无限循环，所以我们需要在某个时候“断开”循环
i  =  0 
for  batch  in  datagen . flow ( x ,  batch_size = 1 ): 
    plt . figure ( i ) 
    imgplot  =  plt . imshow ( image . array_to_img ( batch [ 0 ])) 
    i  +=  1 
    if  i  %  4  ==  0 : 
        break

plt . show ()
```

![Keras 如何训练小数据集](https://img.geek-docs.com/keras/tutorial/xiaoshuju-plt-1.png)
![Keras 如何训练小数据集](https://img.geek-docs.com/keras/tutorial/xiaoshuju-plt-2.png)
![Keras 如何训练小数据集](https://img.geek-docs.com/keras/tutorial/xiaoshuju-plt-3.png)
![Keras 如何训练小数据集](https://img.geek-docs.com/keras/tutorial/xiaoshuju-plt-4.png)
如果使用这种数据增强配置来训练一个新的网络，我们的网络将永远不会看到相同的重覆的输入。然而，看到的输入仍然是彼此关联的，因为它们来自少量的原始图像- 我们不能产生新的信息，我们只能重新混合了现有的信息。密集连接）的分类器之前添加一个Dropout层：

```python
model  =  models . Sequential () 
model . add ( layers . Conv2D ( 32 ,  ( 3 ,  3 ),  activation = 'relu' , 
                        input_shape = ( 150 ,  150 ,  3 ))) 
model . add ( layers . MaxPooling2D (( 2 ,  2 ))) 
model . add ( layers . Conv2D( 64 ,  ( 3 ,  3 ),  activation = 'relu' )) 
model . add ( layers . MaxPooling2D (( 2 ,  2 ))) 
model . add ( layers . Conv2D ( 128 ,  ( 3 ,  3 ),  activation = ' relu' )) 
model . add ( layers . MaxPooling2D (( 2 ,  2))) 
model . add ( layers . Conv2D ( 128 ,  ( 3 ,  3 ),  activation = 'relu' )) 
model . add ( layers . MaxPooling2D (( 2 ,  2 ))) 
model . add ( layers . Flatten () ) 
model . add ( layers . Dropout ( 0.5 )) 
model . add( layers . Dense ( 512 ,  activation = 'relu' )) 
model . add ( layers . Dense ( 1 ,  activation = 'sigmoid' ))

model . compile ( loss = 'binary_crossentropy' , 
              optimizer = optimizers . RMSprop ( lr = 1e-4 ), 
              metrics = [ 'acc' ])
```

我们使用数据扩充（数据扩充）和dropout来训练我们的网络：

```python
train_datagen  =  ImageDataGenerator ( 
    rescale = 1. / 255 , 
    rotation_range = 40 , 
    width_shift_range = 0.2 , 
    height_shift_range = 0.2 , 
    shear_range = 0.2 , 
    zoom_range = 0.2 , 
    horizontal_flip = True ,)

test_datagen  =  ImageDataGenerator ( rescale = 1. / 255 )

train_generator  =  train_datagen . flow_from_directory ( 
        #这是图像资料的目录
        train_dir , 
        #所有的图像大小会被转换成150x150 
        target_size = ( 150 ,  150 ), 
        batch_size = 32 , 
        #由于这是一个二元分类问题, y的lable值也会被转换成二元的标签
        class_mode = 'binary' )

validation_generator  =  test_datagen . flow_from_directory ( 
        validation_dir , 
        target_size = ( 150 ,  150 ), 
        batch_size = 32 , 
        class_mode = 'binary' )

history  =  model . fit_generator ( 
      train_generator , 
      steps_per_epoch = 100 , 
      epochs = 50 , 
      validation_data = validation_generator , 
      validation_steps = 50 )
```

找到属于 2 个类别的 2000 张图像。
找到属于 2 个类别的 1000 张图像。
Epoch 1/50 
100/100 [==============================] – 21s 208ms/step – loss: 0.6937 – acc : 0.5078 – val_loss : 0.6878 – val_acc: 0.4848 
Epoch 2/50 
100/100 [============================] – 19s 189ms/step – loss: 0.6829 – acc: 0.5572 – val_loss : 0.6779 – val_acc: 0.5647 
Epoch 3/50 
100/100 [====================== ========] – 19s 189ms/step – loss: 0.6743 – acc: 0.5816 – val_loss : 0.6520 – val_acc: 0.6136 
Epoch 4/50 
100/100 [============ ==================] – 19s 189ms/step – loss: 0.6619 – acc: 0.6122 – val_loss : 0.6348 – val_acc: 0.6218 
Epoch 5/50
100/100 [==============================] – 19s 189ms/step – loss: 0.6448 – acc: 0.6338 – val_loss : 0.6166 – val_acc: 0.6428 
Epoch 6/50 
100/100 [==============================] – 19s 187ms/step – loss: 0.6342 – acc: 0.6453 – val_loss : 0.6137 – val_acc: 0.6656 
Epoch 7/50 
100/100 [========================== ====] – 19s 189ms/step – loss: 0.6255 – acc: 0.6525 – val_loss : 0.5948 – val_acc: 0.6713 
Epoch 8/50 
100/100 [================ ==============] – 19s 188ms/step – loss: 0.6269 – acc: 0.6444 – val_loss : 0.5899 – val_acc: 0.6891 
Epoch 9/50 
100/100 [====== ========================] – 19s 187ms/step – loss: 0.6105 – acc: 0.6691 – val_loss : 0.6440 – val_acc: 0.6313 
Epoch 10/50
100/100 [==============================] – 19s 189ms/step – loss: 0.5952 – acc: 0.6794 – val_loss : 0.6291 – val_acc: 0.6263 
Epoch 11/50 
100/100 [==============================] – 21s 209ms/step – loss: 0.5926 – acc: 0.6850 – val_loss : 0.5518 – val_acc: 0.7049 
Epoch 12/50 
100/100 [========================== ====] – 19s 193ms/step – loss: 0.5830 – acc: 0.6844 – val_loss : 0.5418 – val_acc: 0.7234 
Epoch 13/50 
100/100 [================ ==============] – 19s 189ms/step – loss: 0.5839 – acc: 0.6903 – val_loss : 0.5382 – val_acc: 0.7354 
Epoch 14/50 
100/100 [====== ========================] – 19s 187ms/step – loss: 0.5663 – acc: 0.6944 – val_loss : 0.5891 – val_acc: 0.6662 
Epoch 15/50
100/100 [==============================] – 19s 187ms/step – loss: 0.5620 – acc: 0.7175 – val_loss : 0.5613 – val_acc: 0.6923 
Epoch 16/50 
100/100 [==============================] – 19s 188ms/step – loss: 0.5458 – acc: 0.7228 – val_loss : 0.4970 – val_acc: 0.7582 
Epoch 17/50 
100/100 [========================== ====] – 19s 187ms/step – loss: 0.5478 – acc: 0.7106 – val_loss : 0.5104 – val_acc: 0.7335 
Epoch 18/50 
100/100 [================ ==============] – 19s 188ms/step – loss: 0.5479 – acc: 0.7250 – val_loss : 0.4990 – val_acc: 0.7544 
Epoch 19/50 
100/100 [====== ========================] – 19s 189ms/step – loss: 0.5390 – acc: 0.7275 – val_loss : 0.4918 – val_acc: 0.7557 
Epoch 20/50
100/100 [==============================] – 19s 187ms/step – loss: 0.5391 – acc: 0.7209 – val_loss : 0.4965 – val_acc: 0.7532 
Epoch 21/50 
100/100 [==============================] – 19s 187ms/step – loss: 0.5379 – acc: 0.7262 – val_loss : 0.4888 – val_acc: 0.7640 
Epoch 22/50 
100/100 [========================== ====] – 19s 188ms/step – loss: 0.5168 – acc: 0.7400 – val_loss : 0.5499 – val_acc: 0.7056 
Epoch 23/50 
100/100 [================ ==============] – 19s 188ms/step – loss: 0.5250 – acc: 0.7369 – val_loss : 0.4768 – val_acc: 0.7697 
Epoch 24/50 
100/100 [====== ========================] – 19s 189ms/step – loss: 0.5088 – acc: 0.7359 – val_loss : 0.4716 – val_acc: 0.7766 
Epoch 25/50
100/100 [==============================] – 19s 188ms/step – loss: 0.5218 – acc: 0.7359 – val_loss : 0.4922 – val_acc: 0.7544 
Epoch 26/50 
100/100 [==============================] – 19s 187ms/step – loss: 0.5143 – acc: 0.7391 – val_loss : 0.4687 – val_acc: 0.7716 
Epoch 27/50 
100/100 [========================== ====] – 19s 188ms/step – loss: 0.5111 – acc: 0.7494 – val_loss : 0.4637 – val_acc: 0.7671 
Epoch 28/50 
100/100 [================ ==============] – 19s 190ms/step – loss: 0.4974 – acc: 0.7506 – val_loss : 0.4899 – val_acc: 0.7557 
Epoch 29/50 
100/100 [====== ========================] – 19s 188ms/step – loss: 0.5136 – acc: 0.7463 – val_loss : 0.5077 – val_acc: 0.7557 
Epoch 30/50
100/100 [==============================] – 19s 190ms/step – loss: 0.5019 – acc: 0.7559 – val_loss : 0.4595 – val_acc: 0.7830 
Epoch 31/50 
100/100 [==============================] – 19s 188ms/step – loss: 0.4961 – acc: 0.7628 – val_loss : 0.4805 – val_acc: 0.7709 
Epoch 32/50 
100/100 [========================== ====] – 19s 190ms/step – loss: 0.4925 – acc: 0.7638 – val_loss : 0.4463 – val_acc: 0.7874 
Epoch 33/50 
100/100 [================ ==============] – 19s 189ms/step – loss: 0.4783 – acc: 0.7700 – val_loss : 0.4667 – val_acc: 0.7824 
Epoch 34/50 
100/100 [====== ========================] – 19s 190ms/step – loss: 0.4792 – acc: 0.7738 – val_loss : 0.4307 – val_acc: 0.8084 
Epoch 35/50
100/100 [==============================] – 19s 190ms/step – loss: 0.4774 – acc: 0.7753 – val_loss : 0.4269 – val_acc: 0.8027 
Epoch 36/50 
100/100 [==============================] – 19s 191ms/step – loss: 0.4756 – acc: 0.7725 – val_loss : 0.4642 – val_acc: 0.7652 
Epoch 37/50 
100/100 [========================== ====] – 19s 190ms/step – loss: 0.4796 – acc: 0.7684 – val_loss : 0.4349 – val_acc: 0.7995 
Epoch 38/50 
100/100 [================ ==============] – 19s 190ms/step – loss: 0.4895 – acc: 0.7665 – val_loss : 0.4588 – val_acc: 0.7836 
Epoch 39/50 
100/100 [====== ========================] – 19s 190ms/step – loss: 0.4832 – acc: 0.7694 – val_loss : 0.4243 – val_acc: 0.8001 
Epoch 40/50
100/100 [==============================] – 19s 191ms/step – loss: 0.4678 – acc: 0.7772 – val_loss : 0.4442 – val_acc: 0.7773 
Epoch 41/50 
100/100 [==============================] – 19s 188ms/step – loss: 0.4623 – acc: 0.7797 – val_loss : 0.4565 – val_acc: 0.7874 
Epoch 42/50 
100/100 [========================== ====] – 19s 190ms/step – loss: 0.4668 – acc: 0.7697 – val_loss : 0.5352 – val_acc: 0.7297 
Epoch 43/50 
100/100 [================ ==============] – 19s 191ms/step – loss: 0.4612 – acc: 0.7906 – val_loss : 0.4236 – val_acc: 0.7951 
Epoch 44/50 
100/100 [====== ========================] – 19s 189ms/step – loss: 0.4598 – acc: 0.7816 – val_loss : 0.4343 – val_acc: 0.7893 
Epoch 45/50
100/100 [==============================] – 19s 189ms/step – loss: 0.4553 – acc: 0.7881 – val_loss : 0.4315 – val_acc: 0.7970 
Epoch 46/50 
100/100 [==============================] – 19s 189ms/step – loss: 0.4621 – acc: 0.7734 – val_loss : 0.4303 – val_acc: 0.8027 
Epoch 47/50 
100/100 [========================== ====] – 19s 189ms/step – loss: 0.4516 – acc: 0.7912 – val_loss : 0.4099 – val_acc: 0.8065 
Epoch 48/50 
100/100 [================ ==============] – 19s 190ms/step – loss: 0.4524 – acc: 0.7822 – val_loss : 0.4088 – val_acc: 0.8115 
Epoch 49/50 
100/100 [====== ========================] – 19s 189ms/step – loss: 0.4508 – acc: 0.7944 – val_loss : 0.4048 – val_acc: 0.8128 
Epoch 50/50
100/100 [==============================] – 19s 188ms/step – loss: 0.4368 – acc: 0.7953 – val_loss : 0.4746 – val_acc: 0.772
我们来保存我们的模型- 我们将在 conv 网络可视化里使用它。

```python
model.save ( 'cats_and_dogs_small_2.h5' )
```

我们再来看我们的结果：

```python
acc  =  history . history [ 'acc' ] 
val_acc  =  history . history [ 'val_acc' ] 
loss  =  history . history [ 'loss' ] 
val_loss  =  history . history [ 'val_loss' ]

epochs  =  range ( len ( acc ))

plt . plot ( epochs ,  acc ,  label = 'Training acc' ) 
plt . plot ( epochs ,  val_acc ,  label = 'Validation acc' ) 
plt . title ( 'Training and validation accuracy' ) 
plt . legend ()

plt . figure ()

plt . plot ( epochs ,  loss ,  label = 'Training loss' ) 
plt . plot ( epochs ,  val_loss ,  label = 'Validation loss' ) 
plt . title ( 'Training and validation loss' ) 
plt . legend ()

plt . show ()
```

![Keras 如何训练小数据集](https://img.geek-docs.com/keras/tutorial/xiaoshuju-3.png)
![Keras 如何训练小数据集](https://img.geek-docs.com/keras/tutorial/xiaoshuju-4.png)

由于数据增加（数据增加）和废弃（辍学）的使用，我们不再有传真训练（过度拟合）的问题：曲线现在大幅度地跟随着曲线验证。我们能够达到82%的准确度，比非非正规化的模型拟改进了15%。

通过进一步利用正规化技术，及调整网络参数（例如每层的数量或网络层数），我们可以更好的准确度，可能高达 86 ~ 87%。训练我们自己的预测网络（convnets），我们证明使用这么少的数据来训练我们要得出一个准确率要高的模型是非常困难的。为了继续提高我们模型对这个问题的准确性，我们将利用训练的模型（预训练模型）来进行操作。

### 6.7总结

在这篇文章中还有一些个人学习到一些有趣的重点：

- 善用**数据扩充**(Data Augmentation) 对训练数据放大的图像可以放大图像
- **Dropout的使用可以抑制过拟合的问题**