---
title: 流畅的Python-03：字典和集合
date: 2024-10-25 12:00:00 +0800
categories: [Python, 流畅的Python]
tags: [编程技术]
---
# 字典和集合

> 字典这个数据结构活跃在所有 Python 程序的背后，即便你的源码里并没有直接用到它。  
> ——A. M. Kuchling 

可散列对象需要实现 `__hash__` 和 `__eq__` 函数。  
如果两个可散列对象是相等的，那么它们的散列值一定是一样的。


```python
# 字典提供了很多种构造方法
a = dict(one=1, two=2, three=3)
b = {'one': 1, 'two': 2, 'three': 3} 
c = dict(zip(['one', 'two', 'three'], [1, 2, 3])) 
d = dict([('two', 2), ('one', 1), ('three', 3)]) 
e = dict({'three': 3, 'one': 1, 'two': 2})
a == b == c == d == e
```


```python
# 字典推导式
r = range(5)
d = {n * 2: n for n in r if n < 3}
print(d)
# setdefault
for n in r:
    d.setdefault(n, 0)
print(d)
```


```python
# defaultdcit & __missing__
class mydefaultdict(dict):
    def __init__(self, value, value_factory):
        super().__init__(value)
        self._value_factory = value_factory

    def __missing__(self, key):
        # 要避免循环调用
        # return self[key]
        self[key] = self._value_factory()
        return self[key]

d = mydefaultdict({1:1}, list)
print(d[1])
print(d[2])
d[3].append(1)
print(d)
```

### 字典的变种
* collections.OrderedDict
* collections.ChainMap (容纳多个不同的映射对象，然后在进行键查找操作时会从前到后逐一查找，直到被找到为止)
* collections.Counter
* colllections.UserDict (dict 的 纯 Python 实现)


```python
# UserDict
# 定制化字典时，尽量继承 UserDict 而不是 dict
from collections import UserDict

class mydict(UserDict):
    def __getitem__(self, key):
        print('Getting key', key)
        return super().__getitem__(key)

d = mydict({1:1})
print(d[1], d[2])
```


```python
# MyppingProxyType 用于构建 Mapping 的只读实例
from types import MappingProxyType

d = {1: 1}
d_proxy = MappingProxyType(d)
print(d_proxy[1])
try:
    d_proxy[1] = 1
except Exception as e:
    print(repr(e))

d[1] = 2
print(d_proxy[1])
```


```python
# set 的操作
# 子集 & 真子集
a, b = {1, 2}, {1, 2}
print(a <= b, a < b)

# discard
a = {1, 2, 3}
a.discard(3)
print(a)

# pop
print(a.pop(), a.pop())
try:
    a.pop()
except Exception as e:
    print(repr(e))
```

### 集合字面量
除空集之外，集合的字面量——`{1}`、`{1, 2}`，等等——看起来跟它的数学形式一模一样。**如果是空集，那么必须写成 `set()` 的形式**，否则它会变成一个 `dict`.  
跟 `list` 一样，字面量句法会比 `set` 构造方法要更快且更易读。

### 集合和字典的实现
集合和字典采用散列表来实现：
1. 先计算 key 的 `hash`, 根据 hash 的某几位（取决于散列表的大小）找到元素后，将该元素与 key 进行比较
2. 若两元素相等，则命中
3. 若两元素不等，则发生散列冲突，使用线性探测再散列法进行下一次查询。

这样导致的后果：
1. 可散列对象必须支持 `hash` 函数；
2. 必须支持 `__eq__` 判断相等性；
3. 若 `a == b`, 则必须有 `hash(a) == hash(b)`。

注：所有由用户自定义的对象都是可散列的，因为他们的散列值由 id() 来获取，而且它们都是不相等的。


### 字典的空间开销
由于字典使用散列表实现，所以字典的空间效率低下。使用 `tuple` 代替 `dict` 可以有效降低空间消费。  
不过：内存太便宜了，不到万不得已也不要开始考虑这种优化方式，**因为优化往往是可维护性的对立面**。

往字典中添加键时，如果有散列表扩张的情况发生，则已有键的顺序也会发生改变。所以，**不应该在迭代字典的过程各种对字典进行更改**。


```python
# 字典中就键的顺序取决于添加顺序

keys = [1, 2, 3]
dict_ = {}
for key in keys:
    dict_[key] = None

for key, dict_key in zip(keys, dict_):
    print(key, dict_key)
    assert key == dict_key

# 字典中键的顺序不会影响字典比较
```
