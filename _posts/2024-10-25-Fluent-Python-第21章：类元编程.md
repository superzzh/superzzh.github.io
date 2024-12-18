---
title: 流畅的Python-21：类元编程
date: 2024-10-25 12:00:00 +0800
categories: [Python, 流畅的Python]
tags: [编程技术]
---
# 类元编程
> （元类）是深奥的知识，99% 的用户都无需关注。如果你想知道是否需要使用元类，我告诉你，不需要（真正需要使用元类的人确信他们需要，无需解释原因）。
> ——Tim Peters, Timsort 算法的发明者，活跃的 Python 贡献者

元类功能强大，但是难以掌握。使用元类可以创建具有某种特质的全新类中，例如我们见过的抽象基类。

本章还会谈及导入时和运行时的区别。

注：除非开发框架，否则不要编写元类。

## 类工厂函数
像 `collections.namedtuple` 一样，类工厂函数的返回值是一个类，这个类的特性（如属性名等）可能由函数参数提供。  
可以参见[官方示例](https://github.com/fluentpython/example-code/blob/master/21-class-metaprog/factories.py)。  
原理：使用 `type` 构造方法，可以构造一个新的类。


```python
help(type)
```

## 元类
元类是一个类，但因为它继承自 `type`，所以可以通过它生成一个类。

在使用元类时，将会调用元类的 `__new__` 和 `__init__`，为类添加更多特性。  
这一步会在**导入**时完成，而不是在类进行实例化时。

元类的作用举例：
* 验证属性
* 一次性把某个/种装饰器依附到多个方法上（记得以前写过一个装饰器来实现这个功能，因为那个类的 `metaclass` 被占了）
* 序列化对象或转换数据
* 对象关系映射
* 基于对象的持久存储
* 动态转换使用其他语言编写的结构

### Python 中各种类的关系
> `object` 类和 `type` 类的关系很独特：`object` 是 `type` 的实例，而 `type` 是 `object` 的子类。
所有类都直接或间接地是 `type` 的实例，不过只有元类同时也是 `type` 的子类，因为**元类从 `type` 类继承了构造类的能力**。

这里面的关系比较复杂，简单理一下
* 实例关系
    * `type` 可以产出类，所以 `type` 的实例是类（`isinstance(int, type)`）；
    * 元类继承自 `type` 类，所以元类也具有生成类实例的**能力**（`isinstance(Sequence, ABCMeta)`)
* 继承关系
    * Python 中一切皆对象，所以所有类都是 `object` 的子类（`object in int.__mro__`）
    * 元类要**继承**自 `type` 对象，所以元类是 `type` 的子类（`type in ABCMeta.__mro__`）


```python
# 构建一个元类
class SomeMeta(type):
    def __init__(cls, name, bases, dic):
        # 这里 __init__ 的是 SomeClass，因为它是个类，所以我们直接用 cls 而不是 self 来命名它
        print('Metaclass __init__')
        # 为我们的类添加一个**类**属性
        cls.a = 1

class SomeClass(metaclass=SomeMeta):
    # 在解释器编译这个类的最后，SomeMeta 的 __init__ 将被调用
    print('Enter SomeClass')
    def __init__(self, val):
        # 这个函数在 SomeClass 实例化时才会被调用
        self.val = val

        
assert SomeClass.a == 1    # 元类为类添加的类属性
sc = SomeClass(2)
assert sc.val == 2
assert sc.a == 1
print(sc.__dict__, SomeClass.__dict__)
```

    Enter SomeClass
    Metaclass __init__
    {'val': 2} {'__module__': '__main__', '__init__': <function SomeClass.__init__ at 0x1113e4510>, '__dict__': <attribute '__dict__' of 'SomeClass' objects>, '__weakref__': <attribute '__weakref__' of 'SomeClass' objects>, '__doc__': None, 'a': 1}


关于类构造过程，可以参见官方 Repo 中的[代码执行练习(evaltime)部分](https://github.com/fluentpython/example-code/tree/master/21-class-metaprog)。


```python
# 用元类构造描述符
from collections import OrderedDict


class Field:
    def __get__(self, instance, cls):
        if instance is None:
            return self
        name = self.__name__
        return instance._value_dict.get(name)

    def __set__(self, instance, val):
        name = self.__name__                    # 通过 _entity_name 属性拿到该字段的名称
        instance._value_dict[name] = val

        
class DesNameMeta(type):
    @classmethod
    def __prepare__(cls, name, bases):
        """
        Python 3 特有的方法，用于返回一个映射对象
        然后传给 __init__ 的 dic 参数
        """
        return OrderedDict()

    def __init__(cls, name, bases, dic):
        field_names = []
        for name, val in dic.items():
            if isinstance(val, Field):
                val.__name__ = name            # 在生成类的时候，将属性名加到了描述符中
                field_names.append(name)
        cls._field_names = field_names
        


class NewDesNameMeta(type):                    # 使用 __new__ 方法构造新类
    def __new__(cls, name, bases, dic):
        for name, val in dic.items():
            if isinstance(val, Field):
                val.__name__ = name
        return super().__new__(cls, name, bases, dic)


class SomeClass(metaclass=DesNameMeta):
    name = Field()
    title = Field()
    
    def __init__(self):
        self._value_dict = {}
    
    def __iter__(self):
        """
        按定义顺序输出属性值
        """
        for field in self._field_names:
            yield getattr(self, field)


assert SomeClass.name.__name__ == 'name'
sc = SomeClass()
sc.name = 'Name'
sc.title = 'Title'
assert sc.name == 'Name'
print(sc._value_dict)
print(list(sc))
```

    {'name': 'Name', 'title': 'Title'}
    ['Name', 'Title']


上面的例子只是演示作用，实际上在设计框架的时候，`SomeClass` 会设计为一个基类（`models.Model`），框架用户只要继承自 `Model` 即可正常使用 `Field` 中的属性名，而无须知道 `DesNameMeta` 的存在。

## 类的一些特殊属性
类上有一些特殊属性，这些属性不会在 `dir()` 中被列出，访问它们可以获得类的一些元信息。  
同时，元类可以更改这些属性，以定制类的某些行为。

* `cls.__bases__`  
    由类的基类组成的元组
* `cls.__qualname__`  
    类或函数的限定名称，即从模块的全局作用域到类的点分路径
* `cls.__subclasses__()`  
    这个方法返回一个列表，包含类的**直接**子类
* `cls.__mro__`  
    类的方法解析顺序，这个属性是只读的，元类无法进行修改
* `cls.mro()`  
    构建类时，如果需要获取储存在类属性 __mro__ 中的超类元组，解释器会调用这个方法。元类可以覆盖这个方法，定制要构建的类解析方法的顺序。

## 延伸阅读
[`types.new_class`](https://docs.python.org/3/library/types.html#types.new_class) 和 `types.prepare_class` 可以辅助我们进行类元编程。

最后附上一个名言警句：

> 此外，不要在生产代码中定义抽象基类(或元类)……如果你很想这样做，我打赌可能是因为你想“找茬”，刚拿到新工具的人都有大干一场的冲动。如果你能避开这些深奥的概念，你(以及未来的代码维护者)的生活将更愉快，因为代码简洁明了。  
> ——Alex Martelli
