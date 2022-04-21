### 1.DataLoader
用于填入数据
```Python
import torch.utils.data as Data
train_loader=Data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True) #shuffle为真表示打乱顺序
```
### 2.将数据转化为pytorch能读取的张量
```Python
import torch
x_train, y_train, x_test, y_test = map(torch.tensor,(x_train, y_train, x_test, y_test))
```
### 3.计算损失函数的库函数
```Python
import torch.nn.functional as F
loss_func = F.cross_entropy
loss = loss_func(model(x), y)
loss.backward() #更新模型梯度、包括weights和bias
```
### 4.nn.Sequential
torch.nn.Sequential是一个Sequential容器，模块将按照构造函数中传递的顺序添加到模块中，也可以传入一个有序模块。使用torch.nn.Sequential添加的激活函数会被会自动加入网络中, 但是 model1中激活函数实际上是在forward()功能中才被调用的。
```Python
import torch
import torch.nn as nn
#model1
class Net(nn.Module):
  def __init__(self,n_feature,n_hidden,n_output):
    super(Net,self).__init__()
    self.hidden=nn.Linear(n_feature,n_hidden)
    self.predict=nn.Linear(n_hidden,n_output)
#model2
class Net(nn.Module):
  def __init__(self, n_feature, n_hidden, n_output):
      super(Net,self).__init__()
      self.net_1 = nn.Sequential(
          nn.Linear(n_feature, n_hidden),
          nn.ReLU(),
          nn.Linear(n_hidden, n_output)
      )

  def forward(self,x):
      x = self.net_1(x)
      return x

model_2 = Net(1,10,1)
print(model_2)

```
### 5.nn.Linear
用于设置网络的全连接层，
```Python
import torch as t
import torch.nn as nn

# in_features由输入张量的形状决定，out_features则决定了输出张量的形状
connected_layer = nn.Linear(in_features = 64*64*3, out_features = 1)

# 假定输入的图像形状为[64,64,3]
input = t.randn(1,64,64,3)

# 将四维张量转换为二维张量之后，才能作为全连接层的输入
input = input.view(1,64*64*3)
print(input.shape)
output = connected_layer(input) # 调用全连接层
print(output.shape)
```
