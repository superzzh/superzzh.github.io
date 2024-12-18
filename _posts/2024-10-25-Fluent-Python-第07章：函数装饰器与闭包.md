---
title: 流畅的Python-07：函数装饰器与闭包
date: 2024-10-25 12:00:00 +0800
categories: [Python, 流畅的Python]
tags: [编程技术]
---
# 函数装饰器与闭包

> 有很多人抱怨，把这个特性命名为“装饰器”不好。主要原因是，这个名称与 GoF 书使用的不一致。**装饰器**这个名称可能更适合在编译器领域使用，因为它会遍历并注解语法书。
> —“PEP 318 — Decorators for Functions and Methods”

本章的最终目标是解释清楚函数装饰器的工作原理，包括最简单的注册装饰器和较复杂的参数化装饰器。  

讨论内容：
* Python 如何计算装饰器语法
* Python 如何判断变量是不是局部的
* 闭包存在的原因和工作原理
* `nonlocal` 能解决什么问题
* 实现行为良好的装饰器
* 标准库中有用的装饰器
* 实现一个参数化的装饰器

装饰器是可调用的对象，其参数是一个函数（被装饰的函数）。装饰器可能会处理被装饰的函数，然后把它返回，或者将其替换成另一个函数或可调用对象。

装饰器两大特性：
1. 能把被装饰的函数替换成其他函数
2. 装饰器在加载模块时立即执行


```python
# 装饰器通常会把函数替换成另一个函数
def decorate(func):
    def wrapped():
        print('Running wrapped()')
    return wrapped

@decorate
def target():
    print('running target()')

target()
# 以上写法等同于
def target():
    print('running target()')

target = decorate(target)
target()
```


```python
# 装饰器在导入时（模块加载时）立即执行
registry = []
def register(func):
    print('running register {}'.format(func))
    registry.append(func)
    return func

@register
def f1():
    print('running f1()')

@register
def f2():
    print('running f2()')


print('registry →', registry)
```

上面的装饰器会原封不动地返回被装饰的函数，而不一定会对函数做修改。  
这种装饰器叫注册装饰器，通过使用它来中心化地注册函数，例如把 URL 模式映射到生成 HTTP 响应的函数上的注册处。

```python
@app.get('/')
def index():
    return "Welcome."
```

可以使用装饰器来实现策略模式，通过它来注册并获取所有的策略。


```python
# 变量作用域规则
b = 1
def f2(a):
    print(a)
    print(b)        # 因为 b 在后面有赋值操作，所以认为 b 为局部变量，所以referenced before assignment
    b = 2

f2(3)
```


```python
# 使用 global 声明 b 为全局变量
b = 1
def f3(a):
    global b
    print(a)
    print(b)
    b = 9

print(b)
f3(2)
print(b)
```


```python
# 闭包
# 涉及嵌套函数时，才会产生闭包问题
def register():
    rrrr = []                # 叫 registry 会跟上面的变量重名掉…
    def wrapped(n):
        print(locals())      # locals() 的作用域延伸到了 wrapped 之外
        rrrr.append(n)
        return rrrr
    return wrapped

# num 为**自由变量**，它未在本地作用域中绑定，但函数可以在其本身的作用域之外引用这个变量
c = register()
print(c(1))
print(c(2))
assert 'rrrr' not in locals()

# 获取函数中的自由变量
print({
    name: cell.cell_contents
    for name, cell in zip(c.__code__.co_freevars, c.__closure__)
})
```

    {'n': 1, 'rrrr': []}
    [1]
    {'n': 2, 'rrrr': [1]}
    [1, 2]
    {'rrrr': [1, 2]}



```python
# 闭包内变量赋值与 nonlocal 声明
def counter():
    n = 0
    def count():
        n += 1      # n = n + 1, 所以将 n 视为局部变量，但未声明，触发 UnboundLocalError
        return n
    return count

def counter():
    n = 0
    def count():
        nonlocal n  # 使用 nonlocal 对 n 进行声明，它可以把 n 标记为局部变量
        n += 1      # 这个 n 和上面的 n 引用的时同一个值，更新这个，上面也会更新
        return n
    return count


c = counter()
print(c(), c())
```

    1 2



```python
# 开始实现装饰器
# 装饰器的典型行为：把被装饰的函数替换成新函数，二者接受相同的参数，而且（通常）返回被装饰的函数本该返回的值，同时还会做些额外操作
import time
from functools import wraps

def clock(func):
    @wraps(func)                         # 用 func 的部分标注属性（如 __doc__, __name__）覆盖新函数的值
    def clocked(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        print(t1 - t0)
        return result
    return clocked

@clock
def snooze(seconds):
    time.sleep(seconds)

snooze(1)
```

    1.002901900000012


Python 内置的三个装饰器分别为 `property`, `classmethod` 和 `staticmethod`.  
但 Python 内置的库中，有两个装饰器很常用，分别为 `functools.lru_cache` 和 [`functools.singledispatch`](https://docs.python.org/3/library/functools.html#functools.singledispatch).


```python
# lru_cache
# 通过内置的 LRU 缓存来存储函数返回值
# 使用它可以对部分递归函数进行优化（比如递归的阶乘函数）（不过也没什么人会这么写吧）
from functools import lru_cache

@lru_cache()
def func(n):
    print(n, 'called')
    return n

print(func(1))
print(func(1))
print(func(2))
```

    1 called
    1
    1
    2 called
    2



```python
# singledispatch
# 单分派泛函数：将多个函数绑定在一起组成一个泛函数，它可以通过参数类型将调用分派至其他函数上
from functools import singledispatch
import numbers

@singledispatch
def func(obj):
    print('Object', obj)

# 只要可能，注册的专门函数应该处理抽象基类，不要处理具体实现（如 int）
@func.register(numbers.Integral)
def _(n):
    print('Integer', n)

# 可以使用函数标注来进行分派注册
@func.register
def _(s:str):
    print('String', s)
    
func(1)
func('test')
func([])
```

    Integer 1
    Object test
    Object []


### 叠放装饰器
```python
@d1
@d2
def func():
    pass

# 等同于
func = d1(d2(func))
```

### 参数化装饰器
为了方便理解，可以把参数化装饰器看成一个函数：这个函数接受任意参数，返回一个装饰器（参数为 func 的另一个函数）。


```python
# 参数化装饰器
def counter(start=1):
    def decorator(func):
        n = start
        def wrapped(*args, **kwargs):
            nonlocal n
            print(f'{func.__name__} called {n} times.')
            n += 1
            return func(*args, **kwargs)
        return wrapped
    return decorator

def test():
    return

t1 = counter(start=1)(test)
t1()
t1()

@counter(start=2)
def t2():
    return

t2()
t2()
```

    test called 1 times.
    test called 2 times.
    t2 called 2 times.
    t2 called 3 times.



```python
# （可能是）更简洁的装饰器实现方式
# 利用 class.__call__

class counter:
    def __init__(self, func):
        self.n = 1
        self.func = func

    def __call__(self, *args, **kwargs):
        print(f'{self.func.__name__} called {self.n} times.')
        self.n += 1
        return self.func(*args, **kwargs)

@counter
def t3():
    return

t3()
t3()
```

    t3 called 1 times.
    t3 called 2 times.



```python
from decorator import decorator

@decorator
def counter(func, *args, **kwargs):
    if not hasattr(func, 'n'):
        func.n = 1
    print(f'{func.__qualname__} called {func.n} times.')
    retval = func(*args, **kwargs)
    func.n += 1
    return retval


@counter
def f(n):
    return n

print(f(2))
print(f(3))
```

    f called 1 times.
    2
    f called 2 times.
    3

