### Lambda
lambda作为一个表达式，定义了一个匿名函数，上例的代码x为入口参数，x+1为函数体。在这里lambda简化了函数定义的书写形式。是代码更为简洁，但是使用函数的定义方式更为直观，易理解。
```Python
func=lambda x:x+1
print(func(1))
#2
```
### map
map()会根据提供的函数对指定序列做映射。第一个参数`function`以参数序列中的每一个元素调用function函数，返回包含每次function函数返回值的新列表。
```Python
foo = [2, 18, 9, 22, 17, 24, 8, 12, 27]
print(list(map(lambda x: x * 2 + 10, foo))) #对foo中所有元素都通过function函数后获得结果，就是一个映射关系
#[14, 46, 28, 54, 44, 58, 26, 34, 64]
```
### reduce
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
### filter
filter函数用于过滤sequence中所有的值，将function返回值为真的值作为结果返回，其余值过滤掉
```Python
foo = [2, 18, 9, 22, 17, 24, 8, 12, 27]
print (list(filter(lambda x: x % 3 == 0, foo)))
#[18, 9, 24, 12, 27]
```
