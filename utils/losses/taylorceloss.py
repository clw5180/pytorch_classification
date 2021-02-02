import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, n=2, ignore_index=-1, reduction='mean', smoothing=0.2):
        super(TaylorCrossEntropyLoss, self).__init__()
        assert n % 2 == 0
        self.taylor_softmax = TaylorSoftmax(dim=1, n=n)
        self.reduction = reduction
        self.ignore_index = ignore_index

        ##### clw modify
        self.label_smoothing = smoothing
        self.class_nums = 5
        smoothing_value = self.label_smoothing / (self.class_nums - 1)
        one_hot = torch.full((self.class_nums,), smoothing_value)
        if self.ignore_index >= 0:
            one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))  # clw note: why

        self.confidence = 1.0 - self.label_smoothing
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
        # log_probs = self.taylor_softmax(logits).log()
        # loss = F.nll_loss(log_probs, labels, reduction=self.reduction, ignore_index=self.ignore_index)

        # clw note: origin modify
        #model_prob = self.one_hot.repeat(labels.size(0), 1).cuda()
        #class_idxs = torch.argmax(labels, dim=1)                           # if target = [[0 1 0 0 0], [0 0 0 1 0], ...]
        #model_prob.scatter_(1, class_idxs.unsqueeze(1), self.confidence)
        #model_prob.scatter_(1, target.unsqueeze(1), self.confidence)      # if target = [1, 3, 4, 2, 0.... 0]

        ### clw modify:因为做了cutmix和 labelsmooth,因此labels并不是整数,
        log_output = self.taylor_softmax(logits).log()
        model_prob = (1 - self.label_smoothing * self.class_nums / (self.class_nums - 1)) * labels + self.label_smoothing / (self.class_nums - 1)

        if self.ignore_index >= 0:
            model_prob.masked_fill_((labels == self.ignore_index).unsqueeze(1), 0)
        # print("model_prob:{}".format(model_prob))
        # print("log_output:{}".format(log_output))

        return -torch.sum(model_prob * log_output) / labels.size(0)

