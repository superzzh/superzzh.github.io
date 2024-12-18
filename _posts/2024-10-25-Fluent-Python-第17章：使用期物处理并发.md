---
title: 流畅的Python-17：使用期物处理并发
date: 2024-10-25 12:00:00 +0800
categories: [Python, 流畅的Python]
tags: [编程技术]
---
# 使用期物处理并发
> 抨击线程的往往是系统程序员，他们考虑的使用场景对一般的应用程序员来说，也许一生都不会遇到……应用程序员遇到的使用场景，99% 的情况下只需知道如何派生一堆独立的线程，然后用队列收集结果。  
> Michele Simionato, 深度思考 Python 的人

本章主要讨论 `concurrent.futures` 模块，并介绍“期物”（future）的概念。

我们在进行 IO 密集型并发编程（如批量下载）时，经常会考虑使用多线程场景来替代依序下载的方案，以提高下载效率。  
在 IO 密集型任务中，如果代码写的正确，那么不管使用哪种并发策略（使用线程或 `asyncio` 包），吞吐量都要比依序执行的代码高很多。

## 期物
期物（Future）表示“**将要**执行并返回结果的任务”，这个概念与 JavaScript 的 `Promise` 对象较为相似。

Python 3.4 起，标准库中有两个 Future 类：`concurrent.futures.Future` 和 `asyncio.Future`。这两个类的作用相同：`Future` 类的实例表示可能已经完成或尚未完成的延迟计算。  
通常情况下自己不应该创建期物或改变期物的状态，而只能由并发框架实例化。  
我们将某个任务交给并发框架后，这个任务将会由框架来进行调度，我们无法改变它的状态，也不能控制计算任务何时结束。


```python
# 简单的期物用法
import time
from concurrent import futures


def fake_download(url):
    time.sleep(1)      # 这里用的是多线程，所以可以直接考虑 sleep
    return url


def download_many(url_list):
    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        to_do = []
        for url in url_list:
            future = executor.submit(fake_download, url)
            to_do.append(future)
            print(f"Scheduled for {url}: {future}")       # 因为 worker 数量有限，所以会有一个 future 处于 pending 状态
        results = []
        for future in futures.as_completed(to_do):
            result = future.result()
            print(f'{future} result: {result}')
            results.append(result)
        return results

download_many(["https://www.baidu.com/", "https://www.google.com/",
               "https://twitter.com/"])
```

    Scheduled for https://www.baidu.com/: <Future at 0x10d4c04a8 state=running>
    Scheduled for https://www.google.com/: <Future at 0x10d4a4f98 state=running>
    Scheduled for https://twitter.com/: <Future at 0x10d4c0198 state=pending>
    <Future at 0x10d4c04a8 state=finished returned str> result: https://www.baidu.com/
    <Future at 0x10d4a4f98 state=finished returned str> result: https://www.google.com/
    <Future at 0x10d4c0198 state=finished returned str> result: https://twitter.com/





    ['https://www.baidu.com/', 'https://www.google.com/', 'https://twitter.com/']



`ThreadExecutor` 使用多线程处理并发。在程序被 IO 阻塞时，Python 标准库会释放 GIL，以允许其它线程运行。  
所以，GIL 的存在并不会对 IO 密集型多线程并发造成太大影响。


`concurrent` 包中提供了 `ThreadPoolExecutor` 和 `ProcessPoolExecutor` 类，分别对应多线程和多进程模型。  
关于两种模型的使用及推荐并发数，我们有一个经验：
* CPU 密集型任务，推荐使用多进程模型，以利用 CPU 的多个核心，`max_workers` 应设置为 CPU 核数；
* IO 密集型任务，多核 CPU 不会提高性能，所以推荐使用多线程模型，可以省下多进程带来的资源开销，`max_workers` 可以尽可能设置多一些。


```python
# Executor.map
# 并发运行多个可调用的对象时，可以使用 map 方法

import time
from concurrent import futures


def fake_download(url):
    time.sleep(1)      # 这里用的是多线程，所以可以直接考虑 sleep
    print(f'[{time.strftime("%H:%M:%S")}] Done with {url}\n', end='')
    return url


def download_many(url_list):
    with futures.ThreadPoolExecutor(max_workers=3) as executor:
        results = executor.map(fake_download, url_list)
        return results

results = download_many(list(range(5)))
print('Results:', list(results))
```

    [15:36:08] Done with 0
    [15:36:08] Done with 1
    [15:36:08] Done with 2
    [15:36:09] Done with 3
    [15:36:09] Done with 4
    Results: [0, 1, 2, 3, 4]


`map` 的使用可能更方便一点，但 `futures.as_completed` 则更灵活：支持不同的运算方法及参数，甚至支持来自不同 `Executor` 的 `future`.

## 总结
15 年的时候看过一篇文章叫[《一行 Python 实现并行化》](https://segmentfault.com/a/1190000000414339)，里面讲述了如何利用 `multiprocessing.Pool.map`（或者 `multiprocessing.dummy.Pool.map`）快速实现多进程 / 多线程模型的并发任务处理。

[`concurrent.furures`](https://docs.python.org/3/library/concurrent.futures.html) 模块于 Python 3.2 版本引入，它把线程、进程和队列是做服务的基础设施，无须手动进行管理即可轻松实现并发任务。同时，这个包引入了“期物”的概念，可以对并发任务更加规范地进行注册及管理。
