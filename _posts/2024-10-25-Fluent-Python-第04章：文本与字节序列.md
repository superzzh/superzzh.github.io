---
title: 流畅的Python-04：文本和字节序列
date: 2024-10-25 12:00:00 +0800
categories: [Python, 流畅的Python]
tags: [编程技术]
---
# 文本和字节序列

> 人类使用文本，计算机使用字节序列  
> —— Esther Nam 和 Travis Fischer  "Character Encoding and Unicode in Python"

Python 3 明确区分了人类可读的文本字符串和原始的字节序列。  
隐式地把字节序列转换成 Unicode 文本（的行为）已成过去。

### 字符与编码
字符的标识，及**码位**，是 0~1114111 的数字，在 Unicode 标准中用 4-6 个十六进制数字表示，如 A 为 U+0041, 高音谱号为 U+1D11E，😂 为 U+1F602.  
字符的具体表述取决于所用的**编码**。编码时在码位与字节序列自减转换时使用的算法。  
把码位转换成字节序列的过程是**编码**，把字节序列转成码位的过程是**解码**。

### 序列类型
Python 内置了两种基本的二进制序列类型：不可变的 `bytes` 和可变的 `bytearray`


```python
# 基本的编码
content = "São Paulo"
for codec in ["utf_8", "utf_16"]:
    print(codec, content.encode(codec))

# UnicodeEncodeError
try:
    content.encode('cp437')
except UnicodeEncodeError as e:
    print(e)

# 忽略无法编码的字符
print(content.encode('cp437', errors='ignore'))
# 把无法编码的字符替换成 ?
print(content.encode('cp437', errors='replace'))
# 把无法编码的字符替换成 xml 实体
print(content.encode('cp437', errors='xmlcharrefreplace'))

# 还可以自己设置错误处理方式
# https://docs.python.org/3/library/codecs.html#codecs.register_error
```

    utf_8 b'S\xc3\xa3o Paulo'
    utf_16 b'\xff\xfeS\x00\xe3\x00o\x00 \x00P\x00a\x00u\x00l\x00o\x00'
    'charmap' codec can't encode character '\xe3' in position 1: character maps to <undefined>
    b'So Paulo'
    b'S?o Paulo'
    b'S&#227;o Paulo'



```python
# 基本的解码
# 处理 UnicodeDecodeError
octets = b'Montr\xe9al'
print(octets.decode('cp1252'))
print(octets.decode('iso8859_7'))
print(octets.decode('koi8_r'))
try:
    print(octets.decode('utf-8'))
except UnicodeDecodeError as e:
    print(e)

# 将错误字符替换成 � (U+FFFD)
octets.decode('utf-8', errors='replace')
```


```python
# Python3 可以使用非 ASCII 名称
São = 'Paulo'
# 但是不能用 Emoji…
```

可以用 `chardet` 检测字符所使用的编码

BOM：字节序标记 (byte-order mark)：  
`\ufffe` 为字节序标记，放在文件开头，UTF-16 用它来表示文本以大端表示(`\xfe\xff`)还是小端表示(`\xff\xfe`)。  
UTF-8 编码并不需要 BOM，但是微软还是给它加了 BOM，非常烦人。

### 处理文本文件
处理文本文件的最佳实践是“三明治”：要尽早地把输入的字节序列解码成字符串，尽量晚地对字符串进行编码输出；在处理逻辑中只处理字符串对象，不应该去编码或解码。  
除非想判断编码，否则不要再二进制模式中打开文本文件；即便如此，也应该使用 `Chardet`，而不是重新发明轮子。  
常规代码只应该使用二进制模式打开二进制文件，比如图像。

### 默认编码
可以使用 `sys.getdefaultincoding()` 获取系统默认编码；  
Linux 的默认编码为 `UTF-8`，Windows 系统中不同语言设置使用的编码也不同，这导致了更多的问题。  
`locale.getpreferredencoding()` 返回的编码是最重要的：这是打开文件的默认编码，也是重定向到文件的 `sys.stdout/stdin/stderr` 的默认编码。不过这个编码在某些系统中是可以改的…  
所以，关于编码默认值的最佳建议是：别依赖默认值。

### Unicode 编码方案
```python
a = 'café'
b = 'cafe\u0301'
print(a, b)                       # café café
print(ascii(a), ascii(b))         # 'caf\xe9' 'cafe\u0301'
print(len(a), len(b), a == b)     # 4 5 False
```

在 Unicode 标准中，é 和 e\u0301 这样的序列叫“标准等价物”，应用程序应将它视为相同的字符。但 Python 看到的是不同的码位序列，因此判断两者不相同。  
我们可以用 `unicodedata.normalize` 将 Unicode 字符串规范化。有四种规范方式：NFC, NFD, NFKC, NFKD

NFC 使用最少的码位构成等价的字符串，而 NFD 会把组合字符分解成基字符和单独的组合字符。  
NFKC 和 NFKD 是出于兼容性考虑，在分解时会将字符替换成“兼容字符”，这种情况下会有格式损失。  
兼容性方案可能会损失或曲解信息（如 "4²" 会被转换成 "42"），但可以为搜索和索引提供便利的中间表述。

> 使用 NFKC 和 NFKC 规范化形式时要小心，而且只能在特殊情况中使用，例如搜索和索引，而不能用户持久存储，因为这两种转换会导致数据损失。


```python
from unicodedata import normalize, name
# Unicode 码位
a = 'café'
b = 'cafe\u0301'
print(a, b)
print(ascii(a), ascii(b))
print(len(a), len(b), a == b)

## NFC 和 NFD
print(len(normalize('NFC', a)), len(normalize('NFC', b)))
print(len(normalize('NFD', a)), len(normalize('NFD', b)))
print(len(normalize('NFC', a)) == len(normalize('NFC', b)))

print('-' * 15)
# NFKC & NFKD
s = '\u00bd'
l = [s, normalize('NFKC', s),  normalize('NFKD', s)]
print(*l)
print(*map(ascii, l))
micro = 'μ'
l = [s, normalize('NFKC', micro)]
print(*l)
print(*map(ascii, l))
print(*map(name, l), sep='; ')
```

### Unicode 数据库
`unicodedata` 库中提供了很多关于 Unicode 的操作及判断功能，比如查看字符名称的 `name`，判断数字大小的 `numric` 等。  
文档见 <https://docs.python.org/3.7/library/unicodedata.html>.


```python
import unicodedata
print(unicodedata.name('½'))
print(unicodedata.numeric('½'), unicodedata.numeric('卅'))
```

    VULGAR FRACTION ONE HALF
    0.5 30.0



```python
# 处理鬼符：按字节序将无法处理的字节序列依序替换成 \udc00 - \udcff 之间的码位
x = 'digits-of-π'
s = x.encode('gb2312')
print(s)                                              # b'digits-of-\xa6\xd0'
ascii_err = s.decode('ascii', 'surrogateescape')
print(ascii_err)                                      # 'digits-of-\udca6\udcd0'
print(ascii_err.encode('ascii', 'surrogateescape'))   # b'digits-of-\xa6\xd0'
```
