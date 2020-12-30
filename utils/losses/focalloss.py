import torch
from torch import nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=2., reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        # print('inputs--', inputs)
        # print('targets--', targets)
        # print('CE_loss--', CE_loss)
        pt = torch.exp(-CE_loss)
        F_loss = ((1 - pt)**self.gamma) * CE_loss
        if self.reduction == 'sum':
            return F_loss.sum()
        elif self.reduction == 'mean':
            return F_loss.mean()


class FocalLoss_clw(nn.Module):
    def __init__(self, gamma=1.0, alpha=0.1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.loss = nn.CrossEntropyLoss(reduction='none')  # TODO: weight=torch.tensor([1.0, 1.0]).cuda()

    def forward(self, inputs, targets):
        pos_mask = (targets == 1)
        neg_mask = (targets == 0)
        CE_loss = self.loss(inputs, targets)
        pt = torch.exp(-CE_loss)
        F_loss = ((1 - pt) ** self.gamma) * CE_loss   # clw modify: TODO
        loss_pos = (1 - self.alpha) * F_loss * pos_mask.float()
        loss_neg = self.alpha * F_loss * neg_mask.float()
        loss = loss_pos + loss_neg
        return loss.mean()


# class FocalLoss_clw_old(nn.Module):
#     def __init__(self, gamma=2.0):
#         super().__init__()
#         self.gamma = gamma
#
#     def forward(self, inputs, targets):
#         #weight_binary = torch.tensor([0.2, 0.8]).cuda()
#         #weight_binary = torch.tensor([0.5, 4]).cuda()
#         #weight_binary = torch.tensor([1.0, 1.0]).cuda()
#         weight_binary = torch.tensor([0.5, 8.0]).cuda()
#         CE_loss = nn.CrossEntropyLoss(reduction='none', weight=weight_binary)(inputs, targets)
#         # print('inputs', inputs)
#         # print('targets', targets)
#         # print('CE_loss', CE_loss)
#         pt = torch.exp(-CE_loss)  # clw note：论文中也给了CE(p, y) = CE(pt) = -log(pt)，反解pt即是该行的公式
#         F_loss = ((1 - pt)**self.gamma) * CE_loss
#         return F_loss.mean()