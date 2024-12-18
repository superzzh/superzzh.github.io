---
title: 流畅的Python-14：可迭代的对象、迭代器和生成器
date: 2024-10-25 12:00:00 +0800
categories: [Python, 流畅的Python]
tags: [编程技术]
---
# 可迭代的对象、迭代器和生成器

> 当我在自己的程序中发现用到了模式，我觉得这就表明某个地方出错了。程序的形式应该仅仅反映它所要解决的问题。代码中其他任何外加的形式都是一个信号，（至少对我来说）表明我对问题的抽象还不够深——这通常意味着自己正在手动完成事情，本应该通过写代码来让宏的扩展自动实现。
> ——Paul Graham, Lisp 黑客和风险投资人

Python 内置了迭代器模式，用于进行**惰性运算**，按需求一次获取一个数据项，避免不必要的提前计算。

迭代器在 Python 中并不是一个具体类型的对象，更多地使指一个具体**协议**。

## 迭代器协议
Python 解释器在迭代一个对象时，会自动调用 `iter(x)`。  
内置的 `iter` 函数会做以下操作：
1. 检查对象是否实现了 `__iter__` 方法（`abc.Iterable`），若实现，且返回的结果是个迭代器（`abc.Iterator`），则调用它，获取迭代器并返回；
2. 若没实现，但实现了 `__getitem__` 方法（`abc.Sequence`），若实现则尝试从 0 开始按顺序获取元素并返回；
3. 以上尝试失败，抛出 `TypeError`，表明对象不可迭代。

判断一个对象是否可迭代，最好的方法不是用 `isinstance` 来判断，而应该直接尝试调用 `iter` 函数。

注：可迭代对象和迭代器不一样。从鸭子类型的角度看，可迭代对象 `Iterable` 要实现 `__iter__`，而迭代器 `Iterator` 要实现 `__next__`. 不过，迭代器上也实现了 `__iter__`，用于[返回自身](https://github.com/python/cpython/blob/3.7/Lib/_collections_abc.py#L268)。

## 迭代器的具体实现
《设计模式：可复用面向对象软件的基础》一书讲解迭代器设计模式时，在“适用性”一 节中说：
迭代器模式可用来：
* 访问一个聚合对象的内容而无需暴露它的内部表示
* 支持对聚合对象的多种遍历
* 为遍历不同的聚合结构提供一个统一的接口（即支持多态迭代）

为了“支持多种遍历”，必须能从同一个可迭代的实例中获取多个**独立**的迭代器，而且各个迭代器要能维护自身的内部状态，因此这一模式正确的实现方式是，每次调用 `iter(my_iterable)` 都新建一个独立的迭代器。这就是为什么这个示例需要定义 `SentenceIterator` 类。

所以，不应该把 Sentence 本身作为一个迭代器，否则每次调用 `iter(sentence)` 时返回的都是自身，就无法进行多次迭代了。


```python
# 通过实现迭代器协议，让一个对象变得可迭代
import re
from collections import abc


class Sentence:
    def  __init__(self, sentence):
        self.sentence = sentence
        self.words = re.findall(r'\w+', sentence)

    def __iter__(self):
        """返回 iter(self) 的结果"""
        return SentenceIterator(self.words)
        

# 推荐的做法是对迭代器对象进行单独实现
class SentenceIterator(abc.Iterator):
    def __init__(self, words):
        self.words = words
        self._index = 0

    def __next__(self):
        """调用时返回下一个对象"""
        try:
            word = self.words[self._index]
        except IndexError:
            raise StopIteration()
        else:
            self._index += 1
        
        return word


    
sentence = Sentence('Return a list of all non-overlapping matches in the string.')
assert isinstance(sentence, abc.Iterable)      # 实现了 __iter__，就支持 Iterable 协议
assert isinstance(iter(sentence), abc.Iterator)
for word in sentence:
    print(word, end='·')
```

    Return·a·list·of·all·non·overlapping·matches·in·the·string·

上面的例子中，我们的 `SentenceIterator` 对象继承自 `abc.Iterator` 通过了迭代器测试。而且 `Iterator` 替我们实现了 `__iter__` 方法。  
但是，如果我们不继承它，我们就需要同时实现 `__next__` 抽象方法和*实际迭代中并不会用到的* `__iter__` 非抽象方法，才能通过 `Iterator` 测试。

## 生成器函数
如果懒得自己写一个迭代器，可以直接用 Python 的生成器函数来在调用 `__iter__` 时生成一个迭代器。

注：在 Python 社区中，大家并没有对“生成器”和“迭代器”两个概念做太多区分，很多人是混着用的。不过无所谓啦。


```python
# 使用生成器函数来帮我们创建迭代器
import re


class Sentence:
    def  __init__(self, sentence):
        self.sentence = sentence
        self.words = re.findall(r'\w+', sentence)

    def __iter__(self):
        for word in self.words:
            yield word
        return

sentence = Sentence('Return a list of all non-overlapping matches in the string.')
for word in sentence:
    print(word, end='·')
```


```python
# 使用 re.finditer 来惰性生成值
# 使用生成器表达式（很久没用过了）
import re


class Sentence:
    def  __init__(self, sentence):
        self.re_word = re.compile(r'\w+')
        self.sentence = sentence

    def __iter__(self):
        return (match.group()
                for match in self.re_word.finditer(self.sentence))

sentence = Sentence('Return a list of all non-overlapping matches in the string.')
for word in sentence:
    print(word, end='·')
```


```python
# 实用模块
import itertools

# takewhile & dropwhile
print(list(itertools.takewhile(lambda x: x < 3, [1, 5, 2, 4, 3])))
print(list(itertools.dropwhile(lambda x: x < 3, [1, 5, 2, 4, 3])))
# zip
print(list(zip(range(5), range(3))))
print(list(itertools.zip_longest(range(5), range(3))))

# itertools.groupby
animals = ['rat', 'bear', 'duck', 'bat', 'eagle', 'shark', 'dolphin', 'lion']
# groupby 需要假定输入的可迭代对象已经按照分组标准进行排序（至少同组的元素要连在一起）
print('----')
for length, animal in itertools.groupby(animals, len):
    print(length, list(animal))
print('----')
animals.sort(key=len)
for length, animal in itertools.groupby(animals, len):
    print(length, list(animal))
print('---')
# tee
g1, g2 = itertools.tee('abc', 2)
print(list(zip(g1, g2)))
```


```python
# 使用 yield from 语句可以在生成器函数中直接迭代一个迭代器
from itertools import chain

def my_itertools_chain(*iterators):
    for iterator in iterators:
        yield from iterator

chain1 = my_itertools_chain([1, 2], [3, 4, 5])
chain2 = chain([1, 2, 3], [4, 5])
print(list(chain1), list(chain2))
```

    [1, 2, 3, 4, 5] [1, 2, 3, 4, 5]


`iter` 函数还有一个鲜为人知的用法：传入两个参数，使用常规的函数或任何可调用的对象创建迭代器。这样使用时，第一个参数必须是可调用的对象，用于不断调用（没有参数），产出各个值；第二个值是哨符，这是个标记值，当可调用的对象返回这个值时，触发迭代器抛出 StopIteration 异常，而不产出哨符。


```python
# iter 的神奇用法
# iter(callable, sentinel)
import random

def rand():
    return random.randint(1, 6)
# 不停调用 rand(), 直到产出一个 5
print(list(iter(rand, 5)))
```
