# Label Protection

使用**pytorch**对*Label Leakage and Protection in Two-party Split Learning*的代码复现，该论文侧重于检测和减轻split learning中的标签泄漏。

论文原文：https://arxiv.org/abs/2102.08504

参考：https://github.com/OscarcarLi/label-protection





## Dependencies

- numpy
- pandas
- torch
- sklearn



## Usage

所有的代码都可以作为 Python 脚本直接运行：

- 标签泄露测试

```sh
python label_leakage.py
```



- 标签泄露防护测试

```sh
python label_leakage_protect.py
```

具体怎么用，去代码里根据注释改（为啥这样写，因为懒 `:)`）



## Overview

该脚本加载信用卡欺诈检测数据集，对其进行预处理，并在数据集上训练两方拆分学习模型。 该脚本还演示了两种类型的攻击：规范攻击和方向攻击，它们试图在拆分学习设置中利用标签泄漏。



## Components

### FirstNet

一个简单的神经网络，具有一个线性层，可将输入特征映射到隐藏维度。

### SecondNet

一个简单的神经网络，具有一个线性层，将隐藏维度映射到单个输出，后跟一个 sigmoid 激活函数。

### torch_auc

根据预测分数计算接受者操作特征曲线下面积 (ROC AUC) 的函数。

### main()

主要功能加载数据集、预处理数据、创建拆分学习模型、训练模型并使用攻击评估模型。





## Dataset

- 为了演示，提供了一小部分 (1%) 的 Criteo 数据集`dataset/criteo`。
- 完整的 Criteo 数据集可以在 [https://www.kaggle.com/c/criteo-display-ad-challenge/data](https://www.kaggle.com/c/criteo-display-ad-challenge/data) 下载





## Training and Attacks

1. 脚本训练split learning模型128个epochs。

2. 训练结束后，它对训练集和测试集都进行规范攻击以评估泄漏。
3. 最后，它对训练集和测试集进行定向攻击以评估泄漏。





## Output

该脚本将输出训练损失、每个时期的 AUC 分数以及对训练集和测试集的攻击的 leak_auc 分数。

- 标签泄露测试

```text
norm_attack: train_leak_auc is  0.9989010989010989
norm_attack: test_leak_auc is  0.9964850615114236
direction_attack: train_leak_auc is  1.0
direction_attack: test_leak_auc is  1.0
```



- 标签泄露防护测试

```python
# 不进行保护
splitnn = SplitNN(clients, optimizers)

# 保护方法1 iso 高斯白噪声
# t 表示高斯噪声的强度
splitnn = ISO_SplitNN(clients, optimizers, t=0.005)

# 保护方法2 max_norm
splitnn = MAX_NORM_SplitNN(clients, optimizers)

# 保护方法3 Marvell
splitnn = Marvell_SplitNN(clients, optimizers)
```

保护方法1 iso 高斯白噪声

```text
epoch=127, loss: 2.113942409154764e-05, auc: 0.979134464513399
norm_attack: train_leak_auc is  0.6034423672481259
norm_attack: test_leak_auc is  0.5907669445323765
direction_attack: train_leak_auc is  0.5040163899514168
direction_attack: test_leak_auc is  0.523174351939784
```

保护方法2 max_norm

```txt
epoch=127, loss: 1.918767210565681e-05, auc: 0.9863095268200492
norm_attack: train_leak_auc is  0.7457103599890079
norm_attack: test_leak_auc is  0.7434289330238456
direction_attack: train_leak_auc is  0.5125663694435032
direction_attack: test_leak_auc is  0.5114908102954405
```

保护方法3 Marvell

```text
epoch=27, loss: 1.9492475777314582e-05, auc: 0.9862081064994371
norm_attack: train_leak_auc is  0.9126043285856598
norm_attack: test_leak_auc is  0.9141042214578572
direction_attack: train_leak_auc is  0.6933941194801713
direction_attack: test_leak_auc is  0.6956137020086595
```

