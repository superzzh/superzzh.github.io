---
title: Dataloader
date: 2024-08-13 09:00:00 +0800
categories: [深度学习与神经网络, Pytorch实用教程]
tags: [机器学习]
math: true
---

# DataLoader特点

1. 支持两种形式数据集读取

    - 映射式(Map-style)：简例中讲解的Dataset类就是映射式，因为它提供了序号到数据的映射（`__getitem__`）。
    - 迭代式(Iterable-style)：编写一个可迭代对象，从中依次获取数据。

2. 自定义采样策略

    DataLoader可借助Sampler自定义采样策略，包括为每个类别设置采样权重以实现1:1的均衡采样。

3. 自动组装成批数据

    mini-batch形式的训练成为了深度学习的标配，如何把数据组装成一个batch数据？DataLoader内部自动实现了该功能，并且可以通过batch_sampler、collate_fn来自定义组装的策略，十分灵活。

4. 多进程数据加载

    通常GPU运算消耗数据会比CPU读取加载数据要快，CPU“生产”跟不上GPU“消费”，因此需要多进程进行加载数据，以满足GPU的消费需求。通常指要设置num_workers 为CPU核心数，如16核的CPU就设置为16。

5. 自动实现锁页内存（Pinning Memory）

# Dataloader简例


```python
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
```


```python
# 以另一个数据集为例
p = Path("mini-hymenoptera_data")

tree_str = ''

def generate_tree(pathname, n=0):
    global tree_str
    if pathname.is_file():
        tree_str += '    |' * n + '-' * 4 + pathname.name + '\n'
    elif pathname.is_dir():
        tree_str += '    |' * n + '-' * 4 + \
            str(pathname.relative_to(pathname.parent)) + '\\' + '\n'
        for cp in pathname.iterdir():
            generate_tree(cp, n + 1)

generate_tree(p)
print(tree_str)
```

    ----mini-hymenoptera_data\
        |----train\
        |    |----ants\
        |    |    |----154124431_65460430f2.jpg
        |    |    |----201790779_527f4c0168.jpg
        |    |    |----226951206_d6bf946504.jpg
        |    |----bees\
        |    |    |----196430254_46bd129ae7.jpg
        |    |    |----473618094_8ffdcab215.jpg
        |----val\
        |    |----ants\
        |    |    |----800px-Meat_eater_ant_qeen_excavating_hole.jpg
        |    |    |----8124241_36b290d372.jpg
        |    |    |----8398478_50ef10c47a.jpg
        |    |----bees\
        |    |    |----54736755_c057723f64.jpg
        |    |    |----72100438_73de9f17af.jpg
    



```python
class AntsBeesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_info = []  # [(path, label), ... , ]
        self.label_array = None
        # 由于标签信息是string，需要一个字典转换为模型训练时用的int类型
        self.str_2_int = {"ants": 0, "bees": 1}
        self._get_img_info()

    def __getitem__(self, index):
        """
        输入标量index, 从硬盘中读取数据，并预处理，to Tensor
        :param index:
        :return:
        """
        path_img, label = self.img_info[index]
        img = Image.open(path_img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(
                self.root_dir))  # 代码具有友好的提示功能，便于debug
        return len(self.img_info)

    def _get_img_info(self):
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith("jpg"):
                    path_img = os.path.join(root, file)
                    sub_dir = os.path.basename(root)
                    label_int = self.str_2_int[sub_dir]
                    self.img_info.append((path_img, label_int))
```


```python
root_dir = r"mini-hymenoptera_data/train"
# =========================== 配合 Dataloader ===================================
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 来自ImageNet数据集统计值
transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

# 训练集总共5个样本
train_set = AntsBeesDataset(root_dir, transform=transforms_train)  # 加入transform
```


```python
# 取数据集，批大小为2，随机打乱
train_loader_bs2 = DataLoader(dataset=train_set, batch_size=2, shuffle=True)
for i, (inputs, target) in enumerate(train_loader_bs2):
    print(i, inputs.shape, target.shape, target)
```

    0 torch.Size([2, 3, 224, 224]) torch.Size([2]) tensor([1, 0])
    1 torch.Size([2, 3, 224, 224]) torch.Size([2]) tensor([1, 0])
    2 torch.Size([1, 3, 224, 224]) torch.Size([1]) tensor([0])



```python
# 取数据集，批大小为3，随机打乱
train_loader_bs3 = DataLoader(dataset=train_set, batch_size=3, shuffle=True)
for i, (inputs, target) in enumerate(train_loader_bs3):
    print(i, inputs.shape, target.shape, target)
```

    0 torch.Size([3, 3, 224, 224]) torch.Size([3]) tensor([1, 0, 1])
    1 torch.Size([2, 3, 224, 224]) torch.Size([2]) tensor([0, 0])



```python
# 取数据集，批大小为2，随机打乱，不足批大小个数舍弃
train_loader_bs2_drop = DataLoader(dataset=train_set, batch_size=2, shuffle=True, drop_last=True)
for i, (inputs, target) in enumerate(train_loader_bs2_drop):
    print(i, inputs.shape, target.shape, target)
```

    0 torch.Size([2, 3, 224, 224]) torch.Size([2]) tensor([1, 0])
    1 torch.Size([2, 3, 224, 224]) torch.Size([2]) tensor([1, 0])


# Dataloader内部原理

1. 初始化dataloader迭代器

   初始化一个dataloder迭代器，获取总共需要多少个batch

2. 对dataloder迭代器进行迭代

   获取一个batch的索引。调用sampler采样器，它也是一个迭代器，获取一批样本索引，若数量达到batch_size，则生成一个batch。

   可以看到，这里生成的batch里面装着的是一批样本索引。

3. 获取信息

   调用fetch函数。首先，调用自定义的dataset，根据样本索引，获取一批样本数据。然后，用collate_fn对数据进行组装。组装的意思是解包再压缩：获取一批样本数据后，再将它们依次拿出合在一起，如一张图片为`(3, 224, 224)`，batch_size是2，则打包后batch样本是`(2, 3, 224, 224)`。这一步通过`torch.stack`实现。
