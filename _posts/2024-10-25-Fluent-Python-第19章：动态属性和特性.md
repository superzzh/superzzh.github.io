---
title: 流畅的Python-19：动态属性和特性
date: 2024-10-25 12:00:00 +0800
categories: [Python, 流畅的Python]
tags: [编程技术]
---
# 动态属性和特性
> 特性至关重要的地方在于，特性的存在使得开发者可以非常安全并且确定可行地将公共数据属性作为类的公共接口的一部分开放出来。
> ——Alex Martelli, Python 贡献者和图书作者

本章内容：
* 特性（property）
* 动态属性存取（`__getattr__` 和 `__setattr__`）
* 对象的动态创建（`__new__`）

## 特性
Python 特性（property）可以使我们在不改变接口的前提下，使用**存取方法**修改数据属性。


```python
# property
class A:
    def __init__(self):
        self._val = 0
    
    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, val):
        print('Set val', val)
        self._val = val
    
    @val.deleter
    def val(self):
        print('Val deleted!')

a = A()
assert a.val == 0
a.val = 1
assert a.val == 1
del a.val
assert a.val == 1      # del val 只是触发了 deleter 方法，再取值时还会执行 val 的 getter 函数
```

    Set val 1
    Val deleted!


## 动态属性
当访问对象的某个属性时，如果对象没有这个属性，解释器会尝试调用对象的 `__attr__` 方法。  
但是注意，这个属性名必须是一个合法的标识符。

注：`__getattr__` 和 `__getattribute__` 的区别在于，`__getattribute__` 在每次访问对象属性时都会触发，而 `__getattr__` 只在该对象没有这个属性的时候才会触发。


```python
class B:
    a = 1
    def __getattr__(self, attr):
        print('getattr', attr)
        return attr

    def __getattribute__(self, attr):
        print('getattribute', attr)
        return super().__getattribute__(attr)

b = B()
print(b.a, b.b)
```

    getattribute a
    getattribute b
    getattr b
    1 b


## __new__ 方法
`__new__` 方法是类上的一个特殊方法，用于生成一个新对象。  
与 `__init__` 不同，`__new__` 方法必须要返回一个对象，而 `__init__` 则不需要。  
调用 `A.__new__` 时，返回的对象不一定需要是 A 类的实例。


```python
# 对象构造过程示意
class Foo:
    # __new__ 是一个特殊方法，所以不需要 @classmethod 装饰器
    def __new__(cls, arg):
        if arg is None:
            return []
        return super().__new__(cls)   # 用 object.__new__ 生成对象后开始执行 __init__ 函数

    def __init__(self, arg):
        print('arg:', arg)
        self.arg = arg


def object_maker(the_class, some_arg):
    new_object = the_class.__new__(the_class, some_arg)
    if isinstance(new_object, the_class):
        the_class.__init__(new_object, some_arg)
    return new_object 
 
# 下述两个语句的作用基本等效
x = Foo('bar')
y = object_maker(Foo, 'bar')
assert x.arg == y.arg == 'bar'
n = Foo(None)
assert n == []
```

    arg bar
    arg bar


## 杂谈
[shelve](https://docs.python.org/3/library/shelve.html) 是 Python 自带的、类 `dict` 的 KV 数据库，支持持久化存储。

```python
import shelve

d = shelve.open(filename)  # open -- file may get suffix added by low-level
                           # library

d[key] = data              # store data at key (overwrites old data if
                           # using an existing key)
data = d[key]              # retrieve a COPY of data at key (raise KeyError
                           # if no such key)
del d[key]                 # delete data stored at key (raises KeyError
                           # if no such key)

flag = key in d            # true if the key exists
klist = list(d.keys())     # a list of all existing keys (slow!)

# as d was opened WITHOUT writeback=True, beware:
d['xx'] = [0, 1, 2]        # this works as expected, but...
d['xx'].append(3)          # *this doesn't!* -- d['xx'] is STILL [0, 1, 2]!

# having opened d without writeback=True, you need to code carefully:
temp = d['xx']             # extracts the copy
temp.append(5)             # mutates the copy
d['xx'] = temp             # stores the copy right back, to persist it

# or, d=shelve.open(filename,writeback=True) would let you just code
# d['xx'].append(5) and have it work as expected, BUT it would also
# consume more memory and make the d.close() operation slower.

d.close()                  # close it
```
架子（shelve）上放一堆泡菜（pickle）坛子…没毛病。
