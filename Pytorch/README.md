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

# in_features由输入张量的形状决定，out_features则决定了输出张量的形状，也即全连接层的神经元个数
connected_layer = nn.Linear(in_features = 64*64*3, out_features = 1)

# 假定输入的图像形状为[64,64,3]
input = t.randn(1,64,64,3)

# 将四维张量转换为二维张量之后，才能作为全连接层的输入
input = input.view(1,64*64*3)
print(input.shape)
output = connected_layer(input) # 调用全连接层
print(output.shape)
```
### 6.nn.BatchNorm1d
用于定义一个归一化函数方法，重要参数是num_features，表示需要归一化的维度，函数的input可以是二维或者三维。当input的维度为(N,C)时，BN将对C维归一化；当input的维度为(N,C,L)或(N,C,L)时，归一化的维度同样为C维，BatchNorm1d会找到第一个维度为C的那一维执行归一化
```Python
BN = nn.BatchNorm1d(100)
input = torch.randn(20, 100)
output = BN(input)
#先定义一个归一化函数BN，需要归一化的维度是100，随后初始化一个矩阵input维度为20*100，最后用BN对矩阵input进行归一化
```
### 7.nn.Conv1d
用于设置1维卷积层：  
`class torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)`  
参数解释如下：  
__in_channels(int)__：输入信号的通道，即输入向量的维度  
__out_channels(int)__：卷积产生的通道，即输出向量的维度  
__kernel_size(int or tuple)__：卷积核的尺寸，卷积核的实际大小为kernel_size*in_channels  
__stride(int or tuple, optional)__：卷积步长  
__padding(int or tuple, optional)__：输入的每条边填0的层数，如果要卷积操作前后向量维度不缩小则padding=(kernel_size-1)/2，或者直接padding="same"
__dilation(int or tuple, optional)__：卷积核元素之间的间距  
__groups(int, optional)__：从输入通道到输出通道的阻塞连接数  
__bias(bool, optional)__：如果bias=True则添加偏置  
输出结果的最后一维的维度d=floor((35+2*padding-kernel_size)/stride)+1=floor((35-2)/2)+1  
![](https://latex.codecogs.com/svg.image?d_{out}=floor(\frac{d_{in}&plus;2\times&space;padding-kernelsize}{stride})&plus;1)
```Python
import torch.nn as nn
conv1=nn.Conv1d(in_channels=256,out_channels=100,kernel_size=2,stride=2,padding=0)
input=torch.randn(32,35,256)  #batch_size=32,句长=35,词向量长度=256
input=input.permute(0,2,1)    #交换第二维和第一维的维度
print(input.size()) #torch.Size([32, 256, 35])
out=conv1(input)    #卷积层是将256维的特征压缩成100维，在最后一维上进行卷积操作
print(out.size())   #torch.Size([32, 100, 17])
```
### 8.permute
将tensor中不同维上的维度换位
```Python
input=torch.randn(32,35,256)
print(input.size()) #torch.Size([32, 35, 256])
input=input.permute(0,2,1)
print(input.size()) #torch.Size([32, 256, 35])
```
### 9.nn.MaxPool1d
用于对输入维度为(N,C,L)的张量，在L维上进行max pooling操作  
`torch.nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)`
参数解释如下:  
__kernel_size__：池化窗口大小  
__stride__：步长，Default value is kernel_size  
__padding__：padding的值，默认就是不padding  
__dilation__：控制扩张的参数  
__return_indices__：if True, will return the max indices along with the outputs  
__ceil_mode__：when True, 会用向上取整而不是向下取整来计算output的shape  
输出的L维大小=(L+ 2*padding-dilation * (kernel_size-1)-1)/stride+1
```Python
m = nn.MaxPool1d(3, stride=2)
input = torch.randn(20, 16, 50)
output = m(input)
print(input.size())   #torch.Size([20, 16, 50])
print(output.size())  #torch.Size([20, 16, 24])
```
