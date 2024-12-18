---
title: 流畅的Python-16：协程
date: 2024-10-25 12:00:00 +0800
categories: [Python, 流畅的Python]
tags: [编程技术]
---
# 协程
> 如果 Python 书籍有一定的指导作用，那么（协程就是）文档最匮乏、最鲜为人知的 Python 特性，因此表面上看是最无用的特性。
> ——David Beazley, Python 图书作者

在“生成器”章节中我们认识了 `yield` 语句。但 `yield` 的作用不只是在生成器运行过程中**返回**一个值，还可以从调用方拿回来一个值（`.send(datum)`），甚至一个异常（`.throw(exc)`）。  
由此依赖，`yield` 语句就成为了一种流程控制工具，使用它可以实现协作式多任务：协程可以把控制器让步给中心调度程序，从而激活其它的协程。

从根本上把 yield 视作控制流程的方式，这样就好理解协程了。

本章涵盖以下话题：
* 生成器作为协程使用时的行为和状态
* 使用装饰器自动预激协程
* 调用方如何使用生成器对象的 `.close()` 和 `.throw(...)` 方法控制协程
* 协程终止时如何返回值
* `yield from` 新句法的用途和语义
* 使用案例——使用协程管理仿真系统中的并发活动

协程的四种状态：
* `GEN_CREATED`: 等待开始执行
* `GEN_RUNNING`: 解释器正在执行
* `GEN_SUSPENDED`: 在 `yield` 表达式处暂停
* `GEN_CLOSED`: 执行结束


```python
# 最简单的协程使用演示
from inspect import getgeneratorstate

def simple_coroutine():
    # GEN_RUNNING 状态
    print("Coroutine started")
    x = yield
    print("Couroutine received:", x)

my_coro = simple_coroutine()
print(getgeneratorstate(my_coro)) # GEN_CREATED
next(my_coro)                     # “预激”(prime)协程，使它能够接收来自外部的值
print(getgeneratorstate(my_coro)) # GEN_SUSPENDED
try:
    my_coro.send(42)
except StopIteration as e:
    print('StopIteration')
print(getgeneratorstate(my_coro)) # GEN_CLOSED
```


```python
# 产出多个值的协程
def async_sum(a=0):
    s = a
    while True:
        n = yield s
        s += n

asum = async_sum()
next(asum)
for i in range(1, 11):
    print(i, asum.send(i))
asum.close()                 # 如果协程不会自己关闭，我们还可以手动终止协程
asum.send(11)
```

    1 1
    2 3
    3 6
    4 10
    5 15
    6 21
    7 28
    8 36
    9 45
    10 55



    ---------------------------------------------------------------------------

    StopIteration                             Traceback (most recent call last)

    <ipython-input-13-0daab1220103> in <module>()
         11     print(i, asum.send(i))
         12 asum.close()                 # 如果协程不会自己关闭，我们还可以手动终止协程
    ---> 13 asum.send(11)
    

    StopIteration: 


协程手动终止的注意事项：  
调用 `gen.closed()` 后，生成器内的 `yield` 语句会抛出 `GeneratorExit` 异常。如果生成器没有处理这个异常，或者抛出了 `StopIteration` 异常（通常是指运行到结尾），调用方不会报错。  
如果收到 `GeneratorExit` 异常，生成器一定不能产出值，否则解释器会抛出 `RuntimeError` 异常。生成器抛出的其他异常会向上冒泡，传给调用方。

协程内异常处理的示例见[官方示例 Repo](https://github.com/fluentpython/example-code/blob/master/16-coroutine/coro_finally_demo.py)。


## yield from
在协程中，`yield from` 语句的主要功能是打开双向通道，把外层的调用方与内层的子生成器连接起来，这样二者可以直接发送和产出值，还可以直接传入异常，而不用在位于中间的协程中添加大量处理异常的样板代码。  
这种夹在中间的生成器，我们称它为“委派生成器”。

子生成器迭代结束后返回（`return`）的值，会交给 `yield from` 函数。

注意：`yield from` 语句会预激生成器，所以与用来预激生成器的装饰器不能放在一起用，否则会出问题。


```python
# 委派生成器
def async_sum(a=0):
    s = a
    while True:
        try:
            n = yield s
        except Exception as e:
            print('Caught exception', e)
            return s
        s += n

def middleware():
    x = yield from async_sum()
    print('Final result:', x)
        
asum = middleware()
next(asum)
for i in range(1, 11):
    print(asum.send(i))

_ = asum.throw(ValueError)
```

    1
    3
    6
    10
    15
    21
    28
    36
    45
    55
    Caught exception 
    Final result: 55



    ---------------------------------------------------------------------------

    StopIteration                             Traceback (most recent call last)

    <ipython-input-16-27762d9b83ae> in <module>()
         19     print(asum.send(i))
         20 
    ---> 21 _ = asum.throw(ValueError)
    

    StopIteration: 


关于“任务式”协程，书中给出了一个[简单的例子](https://github.com/fluentpython/example-code/blob/master/16-coroutine/taxi_sim.py)，用于执行[离散事件仿真](https://zhuanlan.zhihu.com/p/22689081)，仔细研究一下可以对协程有个简单的认识。
