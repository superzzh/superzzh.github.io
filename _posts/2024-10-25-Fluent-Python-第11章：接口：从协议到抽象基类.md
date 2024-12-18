---
title: 流畅的Python-11：接口：从协议到抽象基类
date: 2024-10-25 12:00:00 +0800
categories: [Python, 流畅的Python]
tags: [编程技术]
---
# 接口：从协议到抽象基类
> 抽象类表示接口。  
> ——Bjarne Stroustrup, C++ 之父

本章讨论的话题是接口：从**鸭子类型**的代表特征动态协议，到使接口更明确、能验证实现是否符合规定的抽象基类（Abstract Base Class, ABC）。

> 接口的定义：对象公开方法的子集，让对象在系统中扮演特定的角色。  
> 协议是接口，但不是正式的（只由文档和约定定义），因此协议不能像正式接口那样施加限制。  
> 允许一个类上只实现部分接口。

## 抽象基类（abc）
抽象基类是一个非常实用的功能，可以使用抽象基类来检测某个类是否实现了某种协议，而这个类并不需要继承自这个抽象类。  
[`collections.abc`](https://docs.python.org/3/library/collections.abc.html) 和 [`numbers`](https://docs.python.org/3/library/numbers.html) 模块中提供了许多常用的抽象基类以用于这种检测。

有了这个功能，我们在自己实现函数时，就不需要非常关心外面传进来的参数的**具体类型**（`isinstance(param, list)`），只需要关注这个参数是否支持我们**需要的协议**（`isinstance(param, abc.Sequence`）以保障正常使用就可以了。

但是注意：从 Python 简洁性考虑，最好不要自己创建新的抽象基类，而应尽量考虑使用现有的抽象基类。


```python
# 抽象基类
from collections import abc


class A:
    pass

class B:
    def __len__(self):
        return 0

assert not isinstance(A(), abc.Sized)
assert isinstance(B(), abc.Sized)
assert abc.Sequence not in list.__bases__    # list 并不是 Sequence 的子类
assert isinstance([], abc.Sequence)          # 但是 list 实例支持序列协议
```


```python
# 在抽象基类上进行自己的实现
from collections import abc

class FailedSized(abc.Sized):
    pass


class NormalSized(abc.Sized):
    def __len__(self):
        return 0


n = NormalSized()
print(len(n))
f = FailedSized()       # 基类的抽象协议未实现，Python 会阻止对象实例化
```

    0



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-6-fad553cddcd6> in <module>()
         13 n = NormalSized()
         14 print(len(n))
    ---> 15 f = FailedSized()       # 协议未实现，Python 会阻止对象实例化
    

    TypeError: Can't instantiate abstract class FailedSized with abstract methods __len__


有一点需要注意：抽象基类上的方法并不都是抽象方法。  
换句话说，想继承自抽象基类，只需要实现它上面**所有的抽象方法**即可，有些方法的实现是可选的。  
比如 [`Sequence.__contains__`](https://github.com/python/cpython/blob/3.7/Lib/_collections_abc.py#L889)，Python 对此有自己的实现（使用 `__iter__` 遍历自身，查找是否有相等的元素）。但如果你在 `Sequence` 之上实现的序列是有序的，则可以使用二分查找来覆盖 `__contains__` 方法，从而提高查找效率。

我们可以使用 `__abstractmethods__` 属性来查看某个抽象基类上的抽象方法。这个抽象基类的子类必须实现这些方法，才可以被正常实例化。


```python
# 自己定义一个抽象基类
import abc

# 使用元类的定义方式是 class SomeABC(metaclass=abc.ABCMeta)
class SomeABC(abc.ABC):
    @abc.abstractmethod
    def some_method(self):
        raise NotImplementedError

        
class IllegalClass(SomeABC):
    pass

class LegalClass(SomeABC):
    def some_method(self):
        print('Legal class OK')

    
l = LegalClass()
l.some_method()
il = IllegalClass()    # Raises
```

    Legal class OK



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-22-e78c72cc5a83> in <module>()
         19 l = LegalClass()
         20 l.some_method()
    ---> 21 il = IllegalClass()    # Raises
    

    TypeError: Can't instantiate abstract class IllegalClass with abstract methods some_method


## 虚拟子类
使用 [`register`](https://docs.python.org/3/library/abc.html#abc.ABCMeta.register) 接口可以将某个类注册为某个 ABC 的“虚拟子类”。支持 `register` 直接调用注册，以及使用 `@register` 装饰器方式注册（其实这俩是一回事）。  
注册后，使用 `isinstance` 以及实例化时，解释器将不会对虚拟子类做任何方法检查。  
注意：虚拟子类不是子类，所以虚拟子类不会继承抽象基类的任何方法。


```python
# 虚拟子类
import abc
import traceback

class SomeABC(abc.ABC):
    @abc.abstractmethod
    def some_method(self):
        raise NotImplementedError
    
    def another_method(self):
        print('Another')
    
    @classmethod
    def __subclasshook__(cls, subcls):
        """
        在 register 或者进行 isinstance 判断时进行子类检测
        https://docs.python.org/3/library/abc.html#abc.ABCMeta.__subclasshook__
        """
        print('Subclass:', subcls)
        return True


class IllegalClass:
    pass

SomeABC.register(IllegalClass)                # 注册
il = IllegalClass()
assert isinstance(il, IllegalClass)
assert SomeABC not in IllegalClass.__mro__    # isinstance 会返回 True，但 IllegalClass 并不是 SomeABC 的子类
try:
    il.some_method()                          # 虚拟子类不是子类，不会从抽象基类上继承任何方法
except Exception as e:
    traceback.print_exc()

try:
    il.another_method()
except Exception as e:
    traceback.print_exc()

```

    Subclass: <class '__main__.IllegalClass'>


    Traceback (most recent call last):
      File "<ipython-input-26-d307e70e7794>", line 31, in <module>
        il.some_method()                          # 虚拟子类不是子类，不会从抽象基类上继承任何方法
    AttributeError: 'IllegalClass' object has no attribute 'some_method'
    Traceback (most recent call last):
      File "<ipython-input-26-d307e70e7794>", line 36, in <module>
        il.another_method()
    AttributeError: 'IllegalClass' object has no attribute 'another_method'

