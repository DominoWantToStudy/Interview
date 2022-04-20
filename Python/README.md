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
  key1  key2  data1  data2
0	a	one	-1.672143	1.145812
1	a	two	-0.075998	0.072943
2	b	one	0.933935	0.618680
3	b	two	2.080478	-0.664071
4	a	one	-0.388349	1.283912
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
