---
title: 流畅的Python-12：继承的优缺点
date: 2024-10-25 12:00:00 +0800
categories: [Python, 流畅的Python]
tags: [编程技术]
---
# 继承的优缺点
> （我们）推出继承的初衷是让新手顺利使用只有专家才能设计出来的框架。  
> ——Alan Klay, "The Early History of Smalltalk"

本章探讨继承和子类化，重点是说明对 Python 而言尤为重要的两个细节：
* 子类化内置类型的缺点
* 多重继承和方法解析顺序（MRO）

## 子类化类型的缺点
基本上，内置类型的方法不会调用子类覆盖的方法，所以尽可能不要去子类化内置类型。  
如果有需要使用 `list`, `dict` 等类，`collections` 模块中提供了用于用户继承的 `UserDict`、`userList` 和 `UserString`，这些类经过特殊设计，因此易于扩展。


```python
# 子类化内置类型的缺点
class DoppelDict(dict):
    def __setitem__(self, key, value):
        super().__setitem__(key, [value] * 2)

# 构造方法和 update 都不会调用子类的 __setitem__
dd = DoppelDict(one=1)
dd['two'] = 2
print(dd)
dd.update(three=3)
print(dd)


from collections import UserDict
class DoppelDict2(UserDict):
    def __setitem__(self, key, value):
        super().__setitem__(key, [value] * 2)

# UserDict 中，__setitem__ 对 update 起了作用，但构造函数依然不会调用 __setitem__
dd = DoppelDict2(one=1)
dd['two'] = 2
print(dd)
dd.update(three=3)
print(dd)
```

    {'one': 1, 'two': [2, 2]}
    {'one': 1, 'two': [2, 2], 'three': 3}
    {'one': [1, 1], 'two': [2, 2]}
    {'one': [1, 1], 'two': [2, 2], 'three': [3, 3]}


## 方法解析顺序（Method Resolution Order）
权威论述：[The Python 2.3 Method Resolution Order](https://www.python.org/download/releases/2.3/mro/)  
> Moreover, unless you make strong use of multiple inheritance and you have non-trivial hierarchies, you don't need to understand the C3 algorithm, and you can easily skip this paper. 

Emmm…

OK，提两句：
1. 如果想查看某个类的方法解析顺序，可以访问该类的 `__mro__` 属性；
2. 如果想绕过 MRO 访问某个父类的方法，可以直接调用父类上的非绑定方法。


```python
class A:
    def f(self):
        print('A')


class B(A):
    def f(self):
        print('B')


class C(A):
    def f(self):
        print('C')


class D(B, C):
    def f(self):
        print('D')

    def b_f(self):
        "D -> B"
        super().f()

    def c_f(self):
        "B -> C"
        super(B, self).f()
        # C.f(self)
    
    def a_f(self):
        "C -> A"
        super(C, self).f()


print(D.__mro__)
d = D()
d.f()
d.b_f()
d.c_f()
d.a_f()
```

    (<class '__main__.D'>, <class '__main__.B'>, <class '__main__.C'>, <class '__main__.A'>, <class 'object'>)
    D
    B
    C
    A


## 处理多重继承
书中列出来了一些处理多重继承的建议，以免做出令人费解和脆弱的继承设计：
1. 把接口继承和实现继承区分开  
    如果继承重用的是代码实现细节，通常可以换用组合和委托模式。
2. 使用抽象基类显式表示接口  
    如果基类的作用是定义接口，那就应该定义抽象基类。
3. 通过混入（Mixin）重用代码  
    如果一个类的作用是为多个不相关的子类提供方法实现，从而实现重用，但不体现实际的“上下级”关系，那就应该明确地将这个类定义为**混入类**（Mixin class）。关于 Mixin（我还是习惯英文名），可以看 Python3-Cookbook 的[《利用Mixins扩展类功能》](https://python3-cookbook.readthedocs.io/zh_CN/latest/c08/p18_extending_classes_with_mixins.html)章节。
4. 在名称中明确指出混入  
    在类名里使用 `XXMixin` 写明这个类是一个 Mixin.
5. 抽象基类可以作为混入，反过来则不成立
6. 不要子类化多个具体类  
    在设计子类时，不要在多个**具体基类**上实现多继承。一个子类最好只继承自一个具体基类，其余基类最好应为 Mixin，用于提供增强功能。
7. 为用户提供聚合类  
    如果抽象基类或 Mixin 的组合对客户代码非常有用，那就替客户实现一个包含多继承的聚合类，这样用户可以直接继承自你的聚合类，而不需要再引入 Mixin.
8. “优先使用对象组合，而不是类继承”  
    组合和委托可以代替混入，把行为提供给不同的类，不过这些方法不能取代接口继承去**定义类型层次结构**。

两个实际例子：  
* 很老的 `tkinter` 称为了反例，那个时候的人们还没有充分认识到多重继承的缺点；
* 现代的 `Django` 很好地利用了继承和 Mixin。它提供了非常多的 `View` 类，鼓励用户去使用这些类以免除大量模板代码。
