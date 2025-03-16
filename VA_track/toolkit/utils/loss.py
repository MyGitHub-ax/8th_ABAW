import audmetric
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#自定义你的loss，按照下面的模板
##########################################################
# class template(nn.Module):
    # def __init__(self):
    #     super(template, self).__init__()
    #     self.loss =nn.xxx如果pytorch里面已经有这个loss
    #
    # def forward(self, pred, target):
    #     写上loss如何计算的
    #     return loss


##########################################################
# classification loss
class CELoss(nn.Module):

    def __init__(self):
        super(CELoss, self).__init__()
        self.loss = nn.NLLLoss(reduction='sum')

    def forward(self, pred, target):
        pred = F.log_softmax(pred, 1) # [n_samples, n_classes]
        target = target.long()        # [n_samples]
        loss = self.loss(pred, target) / len(pred)
        return loss

# regression loss
class MSELoss(nn.Module):

    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target):
        #torch.Size([32, 1000, 2])
        pred = pred.view(-1,1)
        target = target.view(-1,1)
        loss = self.loss(pred, target) / len(pred)
        return loss

class CCCLoss(nn.Module):

    def __init__(self):
        super(CCCLoss, self).__init__()

    def forward(self, pred, target):
        arousal_preds = pred[:,:,1]
        valence_preds = pred[:,:, 0]
        arousal_labels = target[:,:, 1]
        valence_labels = target[:,:, 0]
        batch_size, seq_len = arousal_preds.shape

        # 使用列表保存每个视频段的CCC损失
        arousal_losses = [self.ccc_loss(arousal_preds[i], arousal_labels[i]) for i in range(batch_size)]
        valence_losses = [self.ccc_loss(valence_preds[i], valence_labels[i]) for i in range(batch_size)]

        # 将列表转换为张量
        arousal_losses = torch.stack(arousal_losses)
        valence_losses = torch.stack(valence_losses)

        # 计算平均损失
        loss = 2 - (torch.mean(arousal_losses) + torch.mean(valence_losses))

        return loss

    def ccc_loss(self, preds, labels):
        # 计算皮尔逊相关系数
        r = torch.mean((preds - torch.mean(preds)) * (labels - torch.mean(labels))) / (
                torch.std(preds) * torch.std(labels) + 1e-10  # 防止除零错误
        )

        # 计算 CCC
        x_mean = torch.mean(preds)
        y_mean = torch.mean(labels)
        x_std = torch.std(preds)
        y_std = torch.std(labels)
        denominator = (x_std * x_std + y_std * y_std + (x_mean - y_mean) * (x_mean - y_mean) + 1e-10)  # 防止除零错误
        ccc = 2 * r * x_std * y_std / denominator

        return ccc

def get_loss_func(loss_name):
    # 自动检索并返回对应的类
    try:
        # 假设类名和函数名一致（首字母大写）
        loss_class = globals()[f'{loss_name}Loss']
        return loss_class()
    except KeyError:
        raise ValueError(f'Unsupported loss function: {loss_name}')
