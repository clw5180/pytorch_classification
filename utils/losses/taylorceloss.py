import torch
import torch.nn as nn
import torch.nn.functional as F

# implementations reference - https://github.com/CoinCheung/pytorch-loss/blob/master/pytorch_loss/taylor_softmax.py
# paper - https://www.ijcai.org/Proceedings/2020/0305.pdf

class LabelSmoothingLoss(nn.Module):

    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        """Taylor Softmax and log are already applied on the logits"""
        # pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)   # clw note: make onehot true label
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class TaylorSoftmax(nn.Module):
    '''
    This is the autograd version
    '''
    def __init__(self, dim=1, n=2):
        super(TaylorSoftmax, self).__init__()
        assert n % 2 == 0
        self.dim = dim
        self.n = n

    def forward(self, x):
        '''
        usage similar to nn.Softmax:
            >>> mod = TaylorSoftmax(dim=1, n=4)
            >>> inten = torch.randn(1, 32, 64, 64)
            >>> out = mod(inten)
        '''
        fn = torch.ones_like(x)
        denor = 1.
        for i in range(1, self.n+1):
            denor *= i
            fn = fn + x.pow(i) / denor
        out = fn / fn.sum(dim=self.dim, keepdims=True)
        return out


##
# version 1: use torch.autograd
class TaylorCrossEntropyLoss(nn.Module):
    '''
    This is the autograd version
    '''
    def __init__(self,  class_nums, n=2, smoothing=0.2, ignore_index=-1, reduction='mean'):
        super(TaylorCrossEntropyLoss, self).__init__()
        assert n % 2 == 0
        self.taylor_softmax = TaylorSoftmax(dim=1, n=n)
        self.reduction = reduction
        self.ignore_index = ignore_index

        ##### clw modify
        self.lab_smooth = LabelSmoothingLoss(class_nums, smoothing)
        self.label_smoothing = smoothing
        self.class_nums = class_nums

        #####

    def forward(self, logits, labels):
        '''
        usage similar to nn.CrossEntropyLoss:
            >>> crit = TaylorCrossEntropyLoss(n=4)
            >>> inten = torch.randn(1, 10, 64, 64)
            >>> label = torch.randint(0, 10, (1, 64, 64))
            >>> out = crit(inten, label)
        '''
        # clw note: origin is this,
        log_probs = self.taylor_softmax(logits).log()
        if len(labels.shape) != 2:  # clw note: not OneHot
            if self.label_smoothing == 0:
                loss = F.nll_loss(log_probs, labels, reduction=self.reduction, ignore_index=self.ignore_index)
            else:
                loss = self.lab_smooth(log_probs, labels)
        else:
            ############################ clw note: origin modify
            ### clw note: if do mixup or cutmix in dataset, can also use these code
            log_output = self.taylor_softmax(logits).log()
            model_prob = (1 - self.label_smoothing * self.class_nums / (self.class_nums - 1)) * labels + self.label_smoothing / (self.class_nums - 1)

            if self.ignore_index >= 0:
                model_prob.masked_fill_((labels == self.ignore_index).unsqueeze(1), 0)
            # print("model_prob:{}".format(model_prob))
            # print("log_output:{}".format(log_output))
            loss = -torch.sum(model_prob * log_output) / labels.size(0)
        return loss

