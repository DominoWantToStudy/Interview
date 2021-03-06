### 1.梯度消失
由于sigmoid函数求偏导后公式形式的问题，当函数值接近0或1时梯度会非常小，网络很深的情况下模型靠近输入部分的参数很难被更新，模型就训练不起来；可以使用ReLU函数解决这个问题，因此现在很多深度模型往往在隐藏层中使用ReLU而不是Sigmoid（看具体公式推导）。
### 2.随机梯度下降
Motivation: 梯度下降算法中如果数据集较大，则每次迭代中计算损失函数的计算开销都比较大，因此需要随机梯度下降法来解决问题；  
主要思路：计算损失函数时不选用所有训练数据而只随机选取一小部分训练样本，这些小样本称为小批量(Mini-Batch)在，如果迭代次数足够多，小批量就能覆盖完整数据集。
### 3.自适应学习率
主要思路：如果当前处于一个较小的梯度上时，算法选择更大的步长；反之如果当前梯度较大，就给出一个较小的步长；  
Adam是最常见的自适应学习率算法，其计算梯度的一阶动量和二阶动量，进而分别获得一阶动量和二阶动量的滑动平均值，并且有两个因子β1和β2称为梯度的遗忘因子，就是同时考虑上一轮迭代结果的动量与这一轮迭代的梯度，两个β用于调整二者的权重，一般会设置接近1来重点考虑上一轮迭代结果的动量而不直接用当前的梯度来更新参数（这里动量可以理解为已进行的迭代过程获得的梯度的加权平均，越临近的历史梯度权重越大）。
### 4.正则化
把那些使得模型在训练集和测试集上都有很好效果的方法称为正则化方法，包括权重衰减、Dropout和批标准化。
#### 4.1 权重衰减
通过使多项式模型中各项的参数绝对值缩小，则曲线上下摆动的幅度会更小，以此减小过拟合，具体实现方法是给损失函数添加一个参数范式惩戒函数Ω，它的参数就是多项式模型各项的参数θ，并引入参数λ来控制参数范式惩戒的幅度，λ一般是一个较小的值。

![](https://latex.codecogs.com/svg.image?L_{total}=L(y,\widehat{y})&plus;\lambda&space;\Omega&space;(\Theta&space;))

![](https://github.com/DominoWantToStudy/Interview/blob/master/pic/%E5%8F%82%E6%95%B0%E8%8C%83%E5%BC%8F%E6%83%A9%E6%88%92%E5%87%BD%E6%95%B01.PNG)

![](https://github.com/DominoWantToStudy/Interview/blob/master/pic/%E5%8F%82%E6%95%B0%E8%8C%83%E5%BC%8F%E6%83%A9%E6%88%92%E5%87%BD%E6%95%B02.PNG)
#### 4.2 Dropout
当神经元数量过多时，神经网络会出现共适应问题，从而产生过拟合。神经元的共适应指的是神经元之间会互相依赖，一旦有一个神经元失效了，所有依赖它的神经元可能都会失效导致整个网络瘫痪。  
Dropout方法是在训练过程中，将隐藏层的输出按比例随机设置为0，每一层有几个神经元随机地失去与其它层的连接，当然反向传播过程中如果输出为0则对应层的偏导数也会为0，只有还有连接的神经元会被更新。因此Dropout其实是训练很多个参数共享的小网络。  
测试过程中不采用Dropout，所有神经元的输出都不会被设置为0，即所有训练好的小网络一起来做预测，与 __集成学习(Ensemble Learning)__ 类似，EL是训练多个模型来做同一个任务，在测试的时候用所有模型的输出来提高准确率。

![](https://github.com/DominoWantToStudy/Interview/blob/master/pic/Dropout.PNG)
#### 4.3 批标准化
批标准化(Batch Normalization)就是按批对输入做均值为零方差为一的标准化，训练过程中批标准化会通过移动平均法来计算每一批的均值和方差，以此来估算整个训练集的均值和方差，每一批的均值和方差会被用来标准化这批输入，在进行模型测试时，我们也需要保持移动平均值和方差不变来标准化测试集的输入。  
BN可以提高训练的稳定性，并且提升正则化作用，因为移动均值和方差也引入了一定的随机性，因为每一轮训练都是根据当时批次的随机样本来计算均值方差并更新参数的，能够使网络更加鲁棒；另外BN将激活函数的输出从任意的正态分布拉到均值为零方差为一的标准正态分布，使输入落到激活函数的敏感区，即较小的参数变化也会导致较大的loss变化，增大了梯度解决了梯度消失的同时加快了收敛速度。
### 5.RNN
循环神经网络(Recurrent Neural Network, RNN)主要用于处理序列数据，而CNN主要用于处理图像数据。RNN和CNN一样都使用了参数共享，可以让神经网络对序列上不同位置的相同元素重复使用同一组权重。  
RNN深度较大时可能造成梯度爆炸和梯度消失，二者原因相同，当我们把靠近输入的层的梯度通过训练变大时，后面的层的梯度会指数性增大，这就是梯度爆炸。梯度爆炸会引起网络不稳定，最好的结果是无法从训练数据中学习，最坏的结果是出现无法再更新的权重值，即过大大到数据溢出。RNN相当于对每一个输入的单词排成循环相接的多个层，那么如果输入的序列很长，使用简单循环单元的RNN就很可能出现梯度消失或梯度爆炸。  

![](https://github.com/DominoWantToStudy/Interview/blob/master/pic/RNN.PNG)
### 6.LSTM
LSTM网络示意图如下图所示：

![](https://github.com/DominoWantToStudy/Interview/blob/master/pic/LSTM.PNG)  
在LSTM中共有三个基于门的计算机制：遗忘门(Forget Gate)、输入门(Input Gate)、输出门(Output Gate)。遗忘门根据新的输入来决定单元状态中是否有部分信息应该被遗忘；输入门决定哪些输入信息应该被加入单元状态中，目的是长期存储这部分信息并取代被遗忘的信息；输出门根据最新的单元状态，决定LSTM循环单元的输出。
