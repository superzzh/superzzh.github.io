---
title: 流畅的Python-06：使用一等函数实现设计模式
date: 2024-10-25 12:00:00 +0800
categories: [Python, 流畅的Python]
tags: [编程技术]
---
# 使用一等函数实现设计模式

> 符合模式并不表示做得对。
> ——Ralph Johnson: 经典的《设计模式：可复用面向对象软件的基础》的作者之一

本章将对现有的一些设计模式进行简化，从而减少样板代码。

## 策略模式
实现策略模式，可以依赖 `abc.ABC` 和 `abc.abstractmethod` 来构建抽象基类。  
但为了实现“策略”，并不一定要针对每种策略来编写子类，如果需求简单，编写函数就好了。  
我们可以通过 `globals()` 函数来发现所有策略函数，并遍历并应用策略函数来找出最优策略。

## 命令模式
“命令模式”的目的是解耦操作调用的对象（调用者）和提供实现的对象（接受者）。

在 Python 中，我们可以通过定义 `Command` 基类来规范命令调用协议；通过在类上定义 `__call__` 函数，还可以使对象支持直接调用。

```python
import abc

class BaseCommand(ABC):
    def execute(self, *args, **kwargs):
        raise NotImplemented
```

> 事实证明，在 Gamma 等人合著的那本书中，尽管大部分使用 C++ 代码说明（少数使用 Smalltalk），但是 23 个“经典的”设计模式都能很好地在“经典的”Java 中运用。然而，这并不意味着所有模式都能一成不变地在任何语言中运用。
