---
title: 流畅的Python-02：序列构成的数组
date: 2024-10-25 12:00:00 +0800
categories: [Python, 流畅的Python]
tags: [编程技术]
---
# 序列构成的数组
> 你可能注意到了，之前提到的几个操作可以无差别地应用于文本、列表和表格上。  
> 我们把文本、列表和表格叫作数据火车……FOR 命令通常能作用于数据火车上。  
> ——Geurts、Meertens 和 Pemberton  
>   *ABC Programmer’s Handbook*

* 容器序列  
    `list`、`tuple` 和 `collections.deque` 这些序列能存放不同类型的数据。
* 扁平序列  
    `str`、`bytes`、`bytearray`、`memoryview` 和 `array.array`，这类序列只能容纳一种类型。
   
容器序列存放的是它们所包含的任意类型的对象的**引用**，而扁平序列里存放的**是值而不是引用**。换句话说，扁平序列其实是一段连续的内存空间。由此可见扁平序列其实更加紧凑，但是它里面只能存放诸如字符、字节和数值这种基础类型。

序列类型还能按照能否被修改来分类。
* 可变序列  
    `list`、`bytearray`、`array.array`、`collections.deque` 和 `memoryview`。
* 不可变序列  
    `tuple`、`str` 和 `bytes`


```python
# 列表推导式和生成器表达式
symbols = "列表推导式"
[ord(symbol) for symbol in symbols]
(ord(symbol) for symbol in symbols)
```


```python
# 因为 pack/unpack 的存在，元组中的元素会凸显出它们的位置信息
first, *others, last = (1, 2, 3, 4, 5)
print(first, others, last)
# 当然后面很多可迭代对象都支持 unpack 了…
```


```python
# namedtuple
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])
p = Point(1, 2)
print(p, p.x, p.y)
# _asdict() 会返回 OrderedDict
print(p._asdict())
```


```python
# 为什么切片(slice)不返回最后一个元素
a = list(range(6))
# 使用同一个数即可将列表进行分割
print(a[:2], a[2:])
```


```python
# Ellipsis
def test(first, xxx, last):
    print(xxx)
    print(type(xxx))
    print(xxx == ...)
    print(xxx is ...)
    return first, last

# ... 跟 None 一样，有点神奇
print(test(1, ..., 2))
```

### bisect 二分查找


```python
import bisect
def grade(score, breakpoints=[60, 70, 80, 90], grades='FDCBA'):
    i = bisect.bisect(breakpoints, score)
    return grades[i]

print([grade(score) for score in [33, 99, 77, 70, 89, 90, 100]])

a = list(range(0, 100, 10))
# 插入并保持有序
bisect.insort(a, 55)
print(a)
```

### Array
> 虽然列表既灵活又简单，但面对各类需求时，我们可能会有更好的选择。比如，要存放 1000 万个浮点数的话，数组（array）的效率要高得多，因为数组在背后存的并不是 float 对象，而是数字的机器翻译，也就是字节表述。这一点就跟 C 语言中的数组一样。再比如说，如果需要频繁对序列做先进先出的操作，deque（双端队列）的速度应该会更快。

`array.tofile` 和 `fromfile` 可以将数组以二进制格式写入文件，速度要比写入文本文件快很多，文件的体积也小。

> 另外一个快速序列化数字类型的方法是使用 pickle（https://docs.python.org/3/library/pickle.html）模块。pickle.dump 处理浮点数组的速度几乎跟array.tofile 一样快。不过前者可以处理几乎所有的内置数字类型，包含复数、嵌套集合，甚至用户自定义的类。前提是这些类没有什么特别复杂的实现。

array 具有 `type code` 来表示数组类型：具体可见 [array 文档](https://docs.python.org/3/library/array.html).

### memoryview
> memoryview.cast 的概念跟数组模块类似，能用不同的方式读写同一块内存数据，而且内容字节不会随意移动。


```python
import array

arr = array.array('h', [1, 2, 3])
memv_arr = memoryview(arr)
# 把 signed short 的内存使用 char 来呈现
memv_char = memv_arr.cast('B') 
print('Short', memv_arr.tolist())
print('Char', memv_char.tolist())
memv_char[1] = 2  # 更改 array 第一个数的高位字节
# 0x1000000001
print(memv_arr.tolist(), arr)
print('-' * 10)
bytestr = b'123'
# bytes 是不允许更改的
try:
    bytestr[1] = '3'
except TypeError as e:
    print(repr(e))
memv_byte = memoryview(bytestr)
print('Memv_byte', memv_byte.tolist())
# 同样这块内存也是只读的
try:
    memv_byte[1] = 1
except TypeError as e:
    print(repr(e))

```

### Deque
`collections.deque` 是比 `list` 效率更高，且**线程安全**的双向队列实现。

除了 collections 以外，以下 Python 标准库也有对队列的实现：
* queue.Queue (可用于线程间通信)
* multiprocessing.Queue (可用于进程间通信)
* asyncio.Queue
* heapq
