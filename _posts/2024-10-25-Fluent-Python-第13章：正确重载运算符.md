---
title: 流畅的Python-13：正确重载运算符
date: 2024-10-25 12:00:00 +0800
categories: [Python, 流畅的Python]
tags: [编程技术]
---
# 正确重载运算符
> 有些事情让我不安，比如运算符重载。我决定不支持运算符重载，这完全是个个人选择，因为我见过太多 C++ 程序员滥用它。
> ——James Gosling, Java 之父

本章讨论的内容是：
* Python 如何处理中缀运算符（如 `+` 和 `|`）中不同类型的操作数
* 使用鸭子类型或显式类型检查处理不同类型的操作数
* 众多比较运算符（如 `==`、`>`、`<=` 等等）的特殊行为
* 增量赋值运算符（如 `+=`）的默认处理方式和重载方式

重载运算符，如果使用得当，可以让代码更易于阅读和编写。  
Python 出于灵活性、可用性和安全性方面的平衡考虑，对运算符重载做了一些限制：
* 不能重载内置类型的运算符
* 不能新建运算符
* 计算符 `is`、`and`、`or`、`not` 不能重载

Python 算数运算符对应的魔术方法可以见[这里](https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types)。

一个小知识：二元运算符 `+` 和 `-` 对应的魔术方法是 `__add__` 和 `__sub__`，而一元运算符 `+` 和 `-` 对应的魔术方法是 `__pos__` 和 `__neg__`.

## 反向运算符
> 为了支持涉及不同类型的运算，Python 为中缀运算符特殊方法提供了特殊的分派机制。对表达式 `a + b` 来说，解释器会执行以下几步操作。
> 1. 如果 a 有 `__add__` 方法，而且返回值不是 `NotImplemented`，调用 `a.__add__`，然后返回结果。
> 2. 如果 a 没有 `__add__` 方法，或者调用 `__add__` 方法返回 `NotImplemented`，检查 b 有没有 `__radd__` 方法，如果有，而且没有返回 `NotImplemented`，调用 `b.__radd__`，然后返回结果。
> 3. 如果 b 没有 `__radd__` 方法，或者调用 `__radd__` 方法返回 `NotImplemented`，抛出 `TypeError`， 并在错误消息中指明操作数类型不支持。

这样一来，只要运算符两边的任何一个对象正确实现了运算方法，就可以正常实现二元运算操作。

小知识：  
* `NotImplemented` 是一个特殊的单例值，要 `return`；而 `NotImplementedError` 是一个异常，要 `raise`.
* Python 3.5 新引入了 `@` 运算符，用于点积乘法，对应的魔术方法是 [`__matmul__`](https://docs.python.org/3/reference/datamodel.html#object.__matmul__).
* 进行 `!=` 运算时，如果双方对象都没有实现 `__ne__`，解释器会尝试 `__eq__` 操作，并将得到的结果**取反**。

放在这里有点吓人的小知识：
* Python 在进行 `==` 运算时，如果运算符两边的 `__eq__` 都失效了，解释器会用两个对象的 id 做比较\_(:з」∠)\_。_书中用了“最后一搏”这个词…真的有点吓人。_

## 运算符分派
有的时候，运算符另一边的对象可能会出现多种类型：比如对向量做乘法时，另外一个操作数可能是向量，也可能是一个标量。此时，需要在方法实现中，根据操作数的类型进行分派。  
此时有两种选择：
1. 尝试直接运算，如果有问题，捕获 `TypeError` 异常；
2. 在运算前使用 `isinstance` 进行类型判断，在收到可接受类型时在进行运算。  
判断类型时，应进行鸭子类型的判断。应该使用 `isinstance(other, numbers.Integral)`，而不是用 `isinstance(other, int)`，这是之前的知识点。  

不过，在类上定义方法时，是不能用 `functools.singledispatch` 进行单分派的，因为第一个参数是 `self`，而不是 `o`.


```python
# 一个就地运算符的错误示范
class T:
    def __init__(self, s):
        self.s = s

    def __str__(self):
        return self.s

    def __add__(self, o):
        return self.s + o

    def __iadd__(self, o):
        self.s += o
        # 这里必须要返回一个引用，用于传给 += 左边的引用变量
        # return self


t = T('1')
t1 = t
w = t + '2'
print(w, type(w))
t += '2'             # t = t.__iadd__('2')
print(t, type(t))    # t 被我们搞成了 None
print(t1, type(t1))
```

    12 <class 'str'>
    None <class 'NoneType'>
    12 <class '__main__.T'>

