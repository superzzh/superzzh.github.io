---
title: 流畅的Python-20：属性描述符
date: 2024-10-25 12:00:00 +0800
categories: [Python, 流畅的Python]
tags: [编程技术]
---
# 属性描述符
> 学会描述符之后，不仅有更多的工具集可用，还会对 Python 的运作方式有更深入的理解，并由衷赞叹 Python 设计的优雅。  
> ——Raymond Hettinger, Python 核心开发者和专家

本章的话题是描述符。  
描述符是实现了特定协议的类，这个协议包括 `__get__`、`__set__`、和 `__delete__` 方法。

有了它，我们就可以在类上定义一个托管属性，并把所有对实例中托管属性的读写操作交给描述符类去处理。


```python
# 描述符示例：将一个属性托管给一个描述符类
class CharField:                       # 描述符类
    def __init__(self, field_name):
        self.field_name = field_name

    def __get__(self, instance, storage_cls):
        print('__get__', instance, storage_cls)
        if instance is None:            # 直接在类上访问托管属性时，instance 为 None
            return self
        return instance[self.field_name]

    def __set__(self, instance, value):
        print('__set__', instance, value)
        if not isinstance(value, str):
            raise TypeError('Value should be string')
        instance[self.field_name] = value


class SomeModel:                         # 托管类
    name = CharField('name')             # 描述符实例，也是托管类中的托管属性

    def __init__(self, **kwargs):
        self._dict = kwargs              # 出巡属性，用于存储属性

    def __getitem__(self, item):
        return self._dict[item]

    def __setitem__(self, item, value):
        self._dict[item] = value



print(SomeModel.name)
d = SomeModel(name='some name')
print(d.name)
d.name = 'another name'
print(d.name)
try:
    d.name = 1
except Exception as e:
    print(repr(e))
```

    __get__ None <class '__main__.SomeModel'>
    <__main__.CharField object at 0x063AF1F0>
    __get__ <__main__.SomeModel object at 0x063AF4B0> <class '__main__.SomeModel'>
    some name
    __set__ <__main__.SomeModel object at 0x063AF4B0> another name
    __get__ <__main__.SomeModel object at 0x063AF4B0> <class '__main__.SomeModel'>
    another name
    __set__ <__main__.SomeModel object at 0x063AF4B0> 1
    TypeError('Value should be string')


## 描述符的种类
根据描述符上实现的方法类型，我们可以把描述符分为**覆盖型描述符**和**非覆盖型描述符**。

实现 `__set__` 方法的描述符属于覆盖型描述符，因为虽然描述符是类属性，但是实现 `__set__` 方法的话，会覆盖对实例属性的赋值操作。  
而没有实现 `__set__` 方法的描述符是非覆盖型描述符。对实例的托管属性赋值，则会覆盖掉原有的描述符属性，此后再访问该属性时，将不会触发描述符的 `__get__` 操作。如果想恢复原有的描述符行为，则需要用 `del` 把覆盖掉的属性删除。

具体可以看[官方 Repo 的例子](https://github.com/fluentpython/example-code/blob/master/20-descriptor/descriptorkinds.py)。

## 描述符的用法建议
* 如果只想实现一个只读描述符，可以考虑使用 `property` 而不是自己去实现描述符；
* 只读描述符必须有 `__set__` 方法，在方法内抛出 `AttributeError`，防止属性在写时被覆盖；
* 用于验证的描述符可以只有 `__set__` 方法：通过验证后，可以修改 `self.__dict__[key]` 来将属性写入对象；
* 仅有 `__get__` 方法的描述符可以实现高效缓存；
* 非特殊的方法可以被实例属性覆盖。
