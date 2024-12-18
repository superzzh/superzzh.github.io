---
title: 流畅的Python-05：一等函数
date: 2024-10-25 12:00:00 +0800
categories: [Python, 流畅的Python]
tags: [编程技术]
---
# 一等函数

> 不管别人怎么说或怎么想，我从未觉得 Python 受到来自函数式语言的太多影响。我非常熟悉命令式语言，如 C 和 Algol 68，虽然我把函数定为一等对象，但是我并不把 Python 当作函数式编程语言。  
> —— Guido van Rossum: Python 仁慈的独裁者

在 Python 中，函数是一等对象。  
编程语言理论家把“一等对象”定义为满足下述条件的程序实体：
* 在运行时创建
* 能赋值给变量或数据结构中的元素
* 能作为参数传给函数
* 能作为函数的返回结果


```python
# 高阶函数：有了一等函数（作为一等对象的函数），就可以使用函数式风格编程
fruits = ['strawberry', 'fig', 'apple', 'cherry', 'raspberry', 'banana']
print(list(sorted(fruits, key=len)))  # 函数 len 成为了一个参数

# lambda 函数 & map
fact = lambda x: 1 if x == 0 else x * fact(x-1)
print(list(map(fact, range(6))))

# reduce
from functools import reduce
from operator import add
print(reduce(add, range(101)))

# all & any
x = [0, 1]
print(all(x), any(x))
```

    ['fig', 'apple', 'cherry', 'banana', 'raspberry', 'strawberry']
    [1, 1, 2, 6, 24, 120]
    5050
    False True


Python 的可调用对象
* 用户定义的函数：使用 `def` 或 `lambda` 创建
* 内置函数：如 `len` 或 `time.strfttime`
* 内置方法：如 `dict.get`（没懂这俩有什么区别…是说这个函数作为对象属性出现吗？）
* 类：先调用 `__new__` 创建实例，再对实例运行 `__init__` 方法
* 类的实例：如果类上定义了 `__call__` 方法，则实例可以作为函数调用
* 生成器函数：调用生成器函数会返回生成器对象


```python
# 获取函数中的信息
# 仅限关键词参数
def f(a, *, b):
    print(a, b)
f(1, b=2)

# 获取函数的默认参数
# 原生的方法
def f(a, b=1, *, c, d=3):
    pass

def parse_defaults(func):
    code = func.__code__
    argcount = code.co_argcount  # 2
    varnames = code.co_varnames  # ('a', 'b', 'c', 'd')
    argdefaults = dict(zip(reversed(varnames[:argcount]), func.__defaults__))
    kwargdefaults = func.__kwdefaults__
    return argdefaults, kwargdefaults

print(*parse_defaults(f))
print('-----')
# 看起来很麻烦，可以使用 inspect 模块
from inspect import signature
sig = signature(f)
print(str(sig))
for name, param in sig.parameters.items():
    print(param.kind, ':', name, "=", param.default)
print('-----')
# signature.bind 可以在不真正运行函数的情况下进行参数检查
args = sig.bind(1, b=5, c=4)
print(args)
args.apply_defaults()
print(args)
```

    1 2
    {'b': 1} {'d': 3}
    -----
    (a, b=1, *, c, d=3)
    POSITIONAL_OR_KEYWORD : a = <class 'inspect._empty'>
    POSITIONAL_OR_KEYWORD : b = 1
    KEYWORD_ONLY : c = <class 'inspect._empty'>
    KEYWORD_ONLY : d = 3
    -----
    <BoundArguments (a=1, b=5, c=4)>
    <BoundArguments (a=1, b=5, c=4, d=3)>



```python
# 函数注解
def clip(text: str, max_len: 'int > 0'=80) -> str:
    pass

from inspect import signature
sig = signature(clip)
print(sig.return_annotation)
for param in sig.parameters.values():
    note = repr(param.annotation).ljust(13)
    print("{note:13} : {name} = {default}".format(
        note=repr(param.annotation), name=param.name,
        default=param.default))
```

    <class 'str'>
    <class 'str'> : text = <class 'inspect._empty'>
    'int > 0'     : max_len = 80


#### 支持函数式编程的包
`operator` 里有很多函数，对应着 Python 中的内置运算符，使用它们可以避免编写很多无趣的 `lambda` 函数，如：
* `add`: `lambda a, b: a + b`
* `or_`: `lambda a, b: a or b`
* `itemgetter`: `lambda a, b: a[b]`
* `attrgetter`: `lambda a, b: getattr(a, b)`

`functools` 包中提供了一些高阶函数用于函数式编程，如：`reduce` 和 `partial`。  
此外，`functools.wraps` 可以保留函数的一些元信息，在编写装饰器时经常会用到。



```python
# Bonus: 获取闭包中的内容
def fib_generator():
    i, j = 0, 1
    def f():
        nonlocal i, j
        i, j = j, i + j
        return i
    return f

c = fib_generator()
for _ in range(5):
    print(c(), end=' ')
print()
print(dict(zip(
    c.__code__.co_freevars,
    (x.cell_contents for x in c.__closure__))))
```

    1 1 2 3 5 
    {'i': 5, 'j': 8}

