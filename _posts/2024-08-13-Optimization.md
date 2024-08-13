---
title: Optimization
date: 2024-08-13 12:00:00 +0800
categories: [机器学习, Pytorch实用教程]
tags: [机器学习]
math: true
---

# 损失函数

损失函数（loss function）是用来衡量模型输出与真实标签之间的差异，当模型输出越接近标签，认为模型越好，反之亦然。因此，可以得到一个近乎等价的概念，loss越小，模型越好。这样就可以用数值优化的方法不断的让loss变小，即模型的训练。

## L1Loss

`torch.nn.L1Loss(reduction='mean')`

- 功能：计算$\hat{y}$和$y$之间差的绝对值
- reduction：是否需要对loss进行“降维”，这里的reduction指是否将loss值进行取平均（mean）、求和（sum），这样得到的loss值是一个标量。或是保持原尺寸（none），这样返回的是一个同尺寸的tensor。

具体分析`nn.L1Loss`的执行流程:

- `nn.L1Loss`继承自`_Loss`类
- `_Loss`类继承自`nn.Module`类，既然如此，它应当有forward函数
- forward函数当中调用了`F.l1_loss`函数进行计算
- `F.l1_loss`函数底层是C++代码（大多数数值计算通过C++实现）

推而广之，大多数损失函数执行过程如此。

## CrossEntropyLoss

`torch.nn.CrossEntropyLoss(weight=None, ignore_index=- 100, reduction='mean', label_smoothing=0.0) `

交叉熵损失函数，对数似然函数

- `weight`：类别权重，用于调整各类别的损失重要程度，常用于类别不均衡的情况。
- `ignore_index`：忽略某些类别不进行loss计算。
- `label_smoothing`：标签平滑参数。


```python
import torch

y_pred = torch.randn(3, 3) # 标签的预测值
y_true = torch.tensor([2, 0, 1])  # 标签的真实值
print(y_pred)
```

    tensor([[-0.1796, -1.8817, -0.1635],
            [-0.0536, -1.2612, -0.2632],
            [-2.6029, -1.2518, -1.7579]])



```python
# nn.CrossEntropyLoss的计算分三步
## 1. nn.softmax
softmax_output = torch.nn.Softmax(dim=1)(y_pred)
print(softmax_output)
```

    tensor([[0.4548, 0.0829, 0.4622],
            [0.4740, 0.1417, 0.3844],
            [0.1391, 0.5371, 0.3238]])



```python
## 2. nn.log
log_output = torch.log(softmax_output)
print(log_output)
```

    tensor([[-0.7878, -2.4898, -0.7717],
            [-0.7466, -1.9542, -0.9562],
            [-1.9726, -0.6215, -1.1277]])



```python
## 3. nn.NLLLoss
## 第一行预测向量里面选标签为2的；第二行预测向量里面选标签为0的；第三行预测向量里面选标签为1的。加起来除以3取负数。
loss_output = torch.nn.NLLLoss()(log_output, y_true)
print(loss_output)
```

    tensor(0.7133)


# 优化器

有了数据、模型和损失函数，就要选择一个合适的优化器(Optimizer)来优化该模型，使loss不断降低，直到模型收敛。

优化器的实现在`torch.optim`中，在其中有一个核心类是Optimizer，Optimizer在pytorch提供的功能是所有具体优化器的基类，它对优化器进行了抽象与定义，约定了一个优化器应有的功能：

- 获取状态数据
- 加载状态数据
- 梯度清零
- 执行一步优化
- 添加参数组

## 优化器工作方式

- 梯度从哪里来？

  nn.Loss()类计算损失后，调用`.backward()`方法反向传播，计算梯度。

- 更新哪些权重？

  通过loss的反向传播，模型(nn.Module)的权重（Parameter）上有了梯度(.grad)值，但是优化器对哪些权重进行操作呢？

  实际上优化器会对需要操作的权重进行管理，只有被管理的权重，优化器才会对其进行操作。在Optimizer基类中就定义了`add_param_group()`函数来实现参数的管理。通常在实例化的时候，第一个参数就是需要被管理的参数。

- 怎么执行权重更新？

  调用`.step()`方法进行更新，不同的优化方法只需要实现不同的`step()`即可。


```python
loss = loss_f(outputs, labels) # 定义了一个损失函数
loss.backward() # 反向传播，使得每一个paramrter上都有梯度信息
optimizer.step() # 根据paramrter上的梯度信息更新参数
```

解决了一个疑惑：loss和optimizer之间明明没有联系为什么能实现梯度更新？

答案是：loss在反向传播的时候将梯度保留在了parameter上面，optimizer保存了需要更新梯度的参数组，于是它根据参数组参数上面保存的梯度信息实现梯度更新

# 优化器基类Optimizer

## param_group

有时我们需要把参数分组：

- 在finetune过程中，通常让前面层的网络采用较小的学习率，后面几层全连接层采用较大的学习率
- 根据参数类型的不同，例如权值weight，偏置bias，BN的alpha/beta等进行分组管理

参数组是一个list，其元素是一个dict。dict中包含：所管理的参数，对应的超参，例如学习率，`momentum`，`weight_decay`等等。


```python
import torch
import torch.optim as optim

# =================================== 参数组 ========================================
w1 = torch.randn(2, 2)
w1.requires_grad = True

w2 = torch.randn(2, 2)
w2.requires_grad = True

w3 = torch.randn(2, 2)
w3.requires_grad = True

# 一个参数组
optimizer_1 = optim.SGD([w1, w3], lr=0.1)
print('len(optimizer.param_groups): ', len(optimizer_1.param_groups))
print(optimizer_1.param_groups, '\n')

# 两个参数组
optimizer_2 = optim.SGD([{'params': w1, 'lr': 0.1},
                         {'params': w2, 'lr': 0.001}])
print('len(optimizer.param_groups): ', len(optimizer_2.param_groups))
print(optimizer_2.param_groups)
```

    len(optimizer.param_groups):  1
    [{'params': [tensor([[ 1.2612, -0.0059],
            [-0.0988,  1.1554]], requires_grad=True), tensor([[ 0.1868,  1.2103],
            [-1.2627, -0.7893]], requires_grad=True)], 'lr': 0.1, 'momentum': 0, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'maximize': False, 'foreach': None, 'differentiable': False, 'fused': None}] 
    
    len(optimizer.param_groups):  2
    [{'params': [tensor([[ 1.2612, -0.0059],
            [-0.0988,  1.1554]], requires_grad=True)], 'lr': 0.1, 'momentum': 0, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'maximize': False, 'foreach': None, 'differentiable': False, 'fused': None}, {'params': [tensor([[0.6987, 3.1745],
            [0.0827, 0.2612]], requires_grad=True)], 'lr': 0.001, 'momentum': 0, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'maximize': False, 'foreach': None, 'differentiable': False, 'fused': None}]



```python
# =================================== zero_grad ========================================
w1 = torch.randn(2, 2)
w1.requires_grad = True

w2 = torch.randn(2, 2)
w2.requires_grad = True

optimizer = optim.SGD([w1, w2], lr=0.001, momentum=0.9)

optimizer.param_groups[0]['params'][0].grad = torch.randn(2, 2)

print('参数w1的梯度：')
print(optimizer.param_groups[0]['params'][0].grad, '\n')  # 参数组，第一个参数(w1)的梯度

optimizer.zero_grad()
print('执行zero_grad()之后，参数w1的梯度：')
print(optimizer.param_groups[0]['params'][0].grad)  # 参数组，第一个参数(w1)的梯度
# ------------------------- state_dict -------------------------
print('optimizer状态信息：')
print(optimizer.state_dict())
```

    参数w1的梯度：
    tensor([[-1.1581, -1.2463],
            [-1.5890,  1.3832]]) 
    
    执行zero_grad()之后，参数w1的梯度：
    None
    optimizer状态信息：
    {'state': {}, 'param_groups': [{'lr': 0.001, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'maximize': False, 'foreach': None, 'differentiable': False, 'fused': None, 'params': [0, 1]}]}



```python
# =================================== add_param_group ========================================
w1 = torch.randn(2, 2)
w1.requires_grad = True

w2 = torch.randn(2, 2)
w2.requires_grad = True

w3 = torch.randn(2, 2)
w3.requires_grad = True

# 一个参数组
optimizer_1 = optim.SGD([w1, w2], lr=0.1)
print('当前参数组个数: ', len(optimizer_1.param_groups))
print(optimizer_1.param_groups, '\n')

# 增加一个参数组
print('增加一组参数 w3\n')
optimizer_1.add_param_group({'params': w3, 'lr': 0.001, 'momentum': 0.8})

print('当前参数组个数: ', len(optimizer_1.param_groups))
print(optimizer_1.param_groups, '\n')
```

    当前参数组个数:  1
    [{'params': [tensor([[ 0.0504,  0.1076],
            [-1.9668, -0.3113]], requires_grad=True), tensor([[-0.2142,  0.5929],
            [-0.4397, -0.3998]], requires_grad=True)], 'lr': 0.1, 'momentum': 0, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'maximize': False, 'foreach': None, 'differentiable': False, 'fused': None}] 
    
    增加一组参数 w3
    
    当前参数组个数:  2
    [{'params': [tensor([[ 0.0504,  0.1076],
            [-1.9668, -0.3113]], requires_grad=True), tensor([[-0.2142,  0.5929],
            [-0.4397, -0.3998]], requires_grad=True)], 'lr': 0.1, 'momentum': 0, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'maximize': False, 'foreach': None, 'differentiable': False, 'fused': None}, {'params': [tensor([[-1.5862, -1.0751],
            [-0.9665, -0.4928]], requires_grad=True)], 'lr': 0.001, 'momentum': 0.8, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'maximize': False, 'foreach': None, 'differentiable': False, 'fused': None}] 
    


## optimizer其他参数与方法

- `state`

  用于存储优化策略中需要保存的一些缓存值，例如在用momentum时，需要保存之前的梯度，这些数据保存在state中。

- `zero_grad()`

  功能：清零所管理参数的梯度。由于pytorch不会自动清零梯度，因此需要再optimizer中手动清零，然后再执行反向传播，得出当前iteration的loss对权值的梯度。

- `step()`

  功能：执行一步更新，依据当前的梯度进行更新参数。

- `add_param_group(param_group)`

  功能：给optimizer管理的参数组中增加一组参数，可为该组参数定制lr,momentum,weight_decay等，在finetune中常用。

- `state_dict()`

  功能：获取当前state属性。

- `load_state_dict(state_dict)`

  功能：加载所保存的state属性，恢复训练状态。

# 随机梯度下降

## 概念

参数更新公式如下：

$$\theta_{t} = \theta_{t-1} - \gamma \nabla_{\theta}f_t(\theta_{t-1})$$

## 使用


```python
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
optimizer.zero_grad()
loss = loss_f(outputs, labels)
loss.backward()
optimizer.step()
```

- 第一步：实例化`optimizer`
- 第二步：`loss.backward()`之前梯度清零`zero_grad()`
- 第三步：`loss.backward()`之后梯度更新`step()`

# 学习率调整器

深度学习模型训练中调整最频繁的就属学习率了，好的学习率可以使模型逐渐收敛并获得更好的精度。

pytorch实现了自动调整学习率的模块`lr_scheduler`。

`lr_scheduler`的核心属性有：

- `optimizer`：调整器所管理的优化器，优化器中所管理的参数组有对应的学习率，调整器要调整的内容就在那里。
- `base_lrs`：基础学习率，来自于optimizer一开始设定的那个值。
- `last_epoch`：记录迭代次数，通常用于计算下一轮学习率。注意，默认初始值是-1，因为last_epoch的管理逻辑是执行一次，自加1。

核心方法有：

- `state_dict()`和`load_state_dict()`分别是获取调整器的状态数据与加载状态数据。
- `get_last_lr()`和`get_lr()`分别为获取上一次和当前的学习率。
- `print_lr()`是打印学习率。
- `step()`为更新学习率的接口函数，使用者调用 `scheduler.step()`即完成一次更新。

## 使用

- 第一步：实例化`scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=50)`

- 第二步：合适的位置执行`step()`。注意，不同的调整器的更新策略是不一样的，有的是基于epoch维度，有的是基于iteration维度，这个需要注意。


```python
for epoch in range(100):
    # 训练集训练
    model.train()
    for data, labels in train_loader:
        # forward & backward
        outputs = model(data)
        optimizer.zero_grad()

        # loss 计算
        loss = loss_f(outputs, labels)
        loss.backward()
        optimizer.step()

        # 计算分类准确率
        _, predicted = torch.max(outputs.data, 1)
        correct_num = (predicted == labels).sum()
        acc = correct_num / labels.shape[0]
        
    # 学习率调整
    scheduler.step()
```

例如上面的训练流程，每个epoch完成后调整一次学习率。
