### 1.lambda
lambda作为一个表达式，定义了一个匿名函数，上例的代码x为入口参数，x+1为函数体。在这里lambda简化了函数定义的书写形式。是代码更为简洁，但是使用函数的定义方式更为直观，易理解。
```Python
func=lambda x:x+1
print(func(1))
#2
```
### 2.map
map()会根据提供的函数对指定序列做映射。第一个参数`function`以参数序列中的每一个元素调用function函数，返回包含每次function函数返回值的新列表。
```Python
foo = [2, 18, 9, 22, 17, 24, 8, 12, 27]
print(list(map(lambda x: x * 2 + 10, foo))) #对foo中所有元素都通过function函数后获得结果，就是一个映射关系
#[14, 46, 28, 54, 44, 58, 26, 34, 64]
```
### 3.reduce
reduce函数接受一个function和一串sequence，并返回单一的值，以如下方式计算：  
1.初始，function被调用，并传入sequence的前两个items，计算得到result并返回  
2.function继续被调用，并传入上一步中的result，和sequence种下一个item，计算得到result并返回。一直重复这个操作，直到sequence都被遍历完，返回最终结果。  
当sequence只有一个参数的时候reduce()函数会返回它本身！！
```Python
from functools import reduce
foo = [2, 18, 9, 22, 17, 24, 8, 12, 27]
print (reduce(lambda x, y: x + y, foo))
#139
```
### 4.filter
filter函数用于过滤sequence中所有的值，将function返回值为真的值作为结果返回，其余值过滤掉
```Python
foo = [2, 18, 9, 22, 17, 24, 8, 12, 27]
print (list(filter(lambda x: x % 3 == 0, foo)))
#[18, 9, 24, 12, 27]
```
### 5.list与array与DataFrame之间的相互转换
```Python
b=np.array(a)       #list转array
b=a.tolist()        #array转list
b=pd.DataFrame(a)   #array转DataFrame或list转DataFrame，都一样
b=a.values          #DataFrame转array
b=a.values.tolist() #DataFrame转list，其实就是先转array再转list
```
### 6.groupby
groupby用于对DataFrame根据不同的值进行分类，相当于拆成两个表
```Python
import pandas as pd
df = pd.DataFrame({'key1':list('aabba'),
                  'key2': ['one','two','one','two','one'],
                  'data1': np.random.randn(5),
                  'data2': np.random.randn(5)})
print(df)
  key1 key2 data1     data2
0 a    one  -1.672143 1.145812
1 a    two  -0.075998 0.072943
2 b    one  0.933935  0.618680
3 b    two  2.080478  -0.664071
4 a    one  -0.388349 1.283912
for name ,group in df.groupby(['key1']):
    print(name)
    print(group)
a
  key1 key2     data1     data2
0    a  one -1.672143  1.145812
1    a  two -0.075998  0.072943
4    a  one -0.388349  1.283912
b
  key1 key2     data1     data2
2    b  one  0.933935  0.618680
3    b  two  2.080478 -0.664071
```
### 7.apply、iloc
apply就是调用某个函数，同时传入函数所需要的参数  
iloc就是用于DataFrame切片
```Python
print(df.apply(lambda x: x.iloc[2],axis=1))
0   -1.672143
1   -0.075998
2    0.933935
3    2.080478
4   -0.388349
```
### 8.iterrows
iterrows用于对DataFrame进行行遍历
```Python
for index,row in df[:1].iterrows():
    print(index)
    print(row)
0
key1           a
key2         one
data1   -1.67214
data2    1.14581
Name: 0, dtype: object
```
### 9. class
在创建类时，可以为类添加一个特殊的`__init__()`方法，当创建实例时，这个方法会被自动调用为创建的实例添加实例属性。  
`__init__()`方法的第一个参数必须是`self`，后续参数可以自由制定，与定义函数没有区别。

__面试题：__`创建类时，类方法中的self是什么？`  
`self代表类的实例，是通过类创建的实例，在定义这个类时这个实例我们还没有创建，self表示的是我们使用类时创建的那个实例`  

__面试题：__`类属性和实例属性区别？`  
`实例属性在每个实例之间相互独立，而类属性是统一的，如果在类上绑定属性，则所有该类的实例都可以访问这个属性，并且都访问的是同一个东西`  

类的方法是表明这个类用是来做什么，在类的内部，使用`def`关键字来定义方法，与一般函数定义不同，类方法第一个参数必须为self, self代表的是类的实例，其他参数和普通函数完全一样。
  
类还有其他特殊方法又称魔术方法(magic method)，例如`__getitem__`，在定义类时如果希望能按照键取类的值，则需要定义__getitem__方法，实例可以直接返回__getitem__方法执行的结果，当对象是序列时键是整数。当对象是映射时（字典），键可以是任意类型，且特殊方法和init一样不需要手动调用
```Python
class Circle(object):
  pi=3.14 #类属性
  def __init__(self,R):
    self.r=R #这里r是实例属性
  def get_area(self):
    return self.r**2 * Circle.pi  # 通过实例修改pi的值对面积无影响，这个pi为类属性的值
    return self.r**2 * self.pi    # 通过实例修改pi的值对面积我们圆的面积就会改变
  def __getitem__(self,key):
    return "😄" #不管键值多少都会返回笑脸
    return key+self.R #根据键值返回

circle1 = Circle(1)
circle2 = Circle(2)
print('----未修改前-----')
print('pi=\t', Circle.pi)
print('circle1.pi=\t', circle1.pi)  #  3.14
print('circle2.pi=\t', circle2.pi)  #  3.14
print('----通过类名修改后-----')
Circle.pi = 3.14159  # 通过类名修改类属性，所有实例的类属性被改变
print('pi=\t', Circle.pi)   #  3.14159
print('circle1.pi=\t', circle1.pi)   #  3.14159
print('circle2.pi=\t', circle2.pi)   #  3.14159
print('----通过circle1实例名修改后-----')
circle1.pi=3.14111   # 实际上这里是给circle1创建了一个与类属性同名的实例属性
print('pi=\t', Circle.pi)     #  3.14159
print('circle1.pi=\t', circle1.pi)  # 实例属性的访问优先级比类属性高，所以是3.14111   
print('circle2.pi=\t', circle2.pi)  #  3.14159
print('----删除circle1实例属性pi-----')
del circle1.pi
print('pi=\t', Circle.pi) #3.14159
print('circle1.pi=\t', circle1.pi)  #删除实例属性之后就可以访问类属性了，所以是3.14159
print('circle2.pi=\t', circle2.pi)  #3.14159
#所以千万不要在实例上修改类属性，它实际上并没有修改类属性，而是给实例绑定了一个实例属性覆盖了优先级低的类属性
print(circle1[3]) #返回😄或4
```
### 10.view
view是Pytorch中的函数方法，类似于numpy中的reshape方法，就是调整矩阵/张量的形状，注意view前后的元素个数一定要相同！！！可以设定view其中一个参数为-1，代表动态调整这个维度上的元素个数，以保证元素的总数不变
```Python
import torch
v1 = torch.range(1, 16)
v2 = v1.view(4, 4)  #将一个1*16的张量改为4*4，前后都是16
V2 = V1.view(-1,4)  #与上一个语句结果相同
```
