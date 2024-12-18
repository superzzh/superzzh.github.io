---
title: 流畅的Python-10：序列的修改、散列和切片
date: 2024-10-25 12:00:00 +0800
categories: [Python, 流畅的Python]
tags: [编程技术]
---
# 序列的修改、散列和切片
> 不要检查它**是不是**鸭子、它的**叫声**像不像鸭子、它的**走路姿势**像不像鸭子，等等。具体检查什么取决于你想用语言的哪些行为。(comp.lang.python, 2000 年 7 月 26 日)  
> ——Alex Martelli

本章在 `Vector2d` 基础上进行改进以支持多维向量。不过我不想写那么多 `Vector` 代码了，所以我会尝试对里面讲到的知识进行一些抽象。

当然，如果有兴趣也可以研究一下书中实现的[多维向量代码](https://github.com/fluentpython/example-code/tree/master/10-seq-hacking)。

## Python 序列协议
鸭子类型（duck typing）：“如果一个东西长的像鸭子、叫声像鸭子、走路像鸭子，那我们可以认为它就是鸭子。”

Python 中，如果我们在类上实现了 `__len__` 和 `__getitem__` 接口，我们就可以把它用在任何期待序列的场景中。


```python
# 序列协议中的切片
class SomeSeq:
    def __init__(self, seq):
        self._seq = list(seq)
    
    def __len__(self):
        return len(self._seq)

    def __getitem__(self, index):
        if isinstance(index, slice):
            # 专门对 slice 做一个自己的实现
            start, stop = index.start, index.stop
            step = index.step
            
            if start is None:
                start = 0
            elif start < 0:
                start = len(self) + start
            else:
                start = min(start, len(self))
            if stop is None:
                stop = len(self)
            elif stop < 0:
                stop = len(self) + stop
            else:
                stop = min(stop, len(self))
            if step is None:
                step = 1
            elif step == 0:
                raise ValueError("slice step cannot be zero")

            # 以上的复杂逻辑可以直接使用 slice 的接口
            # start, stop, step = index.indices(len(self))
            index_range = range(start, stop, step)
            return [self._seq[i] for i in index_range]
        else:
            return self._seq[index]

        
seq = SomeSeq([1, 2, 3, 4, 5])
print(seq[2])
print(seq[2:4], seq[:5], seq[:5:2], seq[3:], seq[:200])
print(seq[:-1], seq[-1:-5:-1])
```

    3
    [3, 4] [1, 2, 3, 4, 5] [1, 3, 5] [4, 5] [1, 2, 3, 4, 5]
    [1, 2, 3, 4] [5, 4, 3, 2]



```python
# __getitem__ 的参数不一定是单个值或者 slice，还有可能是 tuple
class SomeSeq:
    def __init__(self, seq):
        self._seq = list(seq)
    
    def __len__(self):
        return len(self._seq)

    def __getitem__(self, item):
        return item

seq = SomeSeq([1, 2, 3, 4, 5])
print(seq[1, 2, 3])
print(seq[:5, 2:5:2, -1:5:3])
```

    (1, 2, 3)
    (slice(None, 5, None), slice(2, 5, 2), slice(-1, 5, 3))



```python
# zip 和 enumerate: 知道这两个方法可以简化一些场景
l1, l2 = [1, 2, 3], [1, 2, 3, 4, 5]
for n1, n2 in zip(l1, l2):
    print(n1, n2)

print('---')
# 不要这么写
# for index in range(len(l1)):
#     print(index, l1[index])
for index, obj in enumerate(l1):
    print(index, obj)

print('---')
# list 分组的快捷操作
# 注意：只用于序列长度是分组的倍数的场景，否则最后一组会丢失
l = [1,2,3,4,5,6,7,8,9]
print(list(zip(*[iter(l)] * 3)))
```

    1 1
    2 2
    3 3
    ---
    0 1
    1 2
    2 3
    ---
    [(1, 2, 3), (4, 5, 6), (7, 8, 9)]

