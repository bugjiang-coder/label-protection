import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def norm_attack(splitnn, dataloader, attack_criterion, device="cpu"):
    epoch_labels = []
    epoch_g_norm = []
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        for opt in splitnn.optimizers:
            # 第一次训练需要清空梯度
            opt.zero_grad()

        # 输出第一个模型的结果
        outputs = splitnn(inputs)
        loss = attack_criterion(outputs, labels)

        # iso和max_norm使用
        splitnn.backward(loss)
        # 对应marvell使用下面
        # splitnn.backward(loss, labels)

        grad_from_server = splitnn.clients[0].grad_from_next_client
        # 注pow(2)是对每个元素平方
        # sum是元素相加,dim表示横向相加,也就是128个16维梯度加为128个单维
        # sqrt开平方，综上表示取模
        g_norm = grad_from_server.pow(2).sum(dim=1).sqrt()
        epoch_labels.append(labels)
        epoch_g_norm.append(g_norm)

    epoch_labels = torch.cat(epoch_labels)
    epoch_g_norm = torch.cat(epoch_g_norm)

    score = roc_auc_score(epoch_labels.cpu(), epoch_g_norm.cpu().view(-1, 1))
    return score


def direction_attack(splitnn, dataloader, attack_criterion, device="cpu"):
    epoch_labels = []
    epoch_g_direc = []
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        for opt in splitnn.optimizers:
            # 第一次训练需要清空梯度
            opt.zero_grad()

        # 输出第一个模型的结果
        outputs = splitnn(inputs)
        # print(outputs.shape)
        loss = attack_criterion(outputs, labels)

        # iso和max_norm使用
        splitnn.backward(loss)
        # 对应marvell使用下面
        # splitnn.backward(loss, labels)

        grad_from_server = splitnn.clients[0].grad_from_next_client

        g_direc = torch.split(grad_from_server, 1, 0)

        # 余弦相似度类
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        if labels[0].item() > 0.5:
            sig_g_direc = torch.ones([1])
            for i in range(1, len(g_direc)):
                if cos(g_direc[0], g_direc[i]).item() >= 0:
                    sig_g_direc = torch.cat((sig_g_direc, torch.ones([1])), 0)
                else:
                    sig_g_direc = torch.cat((sig_g_direc, torch.zeros([1])), 0)
        else:
            sig_g_direc = torch.zeros([1])
            for i in range(1, len(g_direc)):
                if cos(g_direc[0], g_direc[i]).item() < 0:
                    sig_g_direc = torch.cat((sig_g_direc, torch.ones([1])), 0)
                else:
                    sig_g_direc = torch.cat((sig_g_direc, torch.zeros([1])), 0)

        epoch_labels.append(labels)
        epoch_g_direc.append(sig_g_direc)

    epoch_labels = torch.cat(epoch_labels)
    epoch_g_direc = torch.cat(epoch_g_direc)

    score = roc_auc_score(epoch_labels.cpu(), epoch_g_direc.cpu().view(-1, 1))
    return score
