import torch
from torch import nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):  # clw note: 标签无论是单独的0 1 2 3,还是onehot类型,都可以用,这里已经做了兼容  if len(target.shape) == 2
    def __init__(self, class_nums, label_smoothing, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (class_nums - 1)
        one_hot = torch.full((class_nums,), smoothing_value)
        if self.ignore_index >= 0:
            one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """

        log_output = F.log_softmax(output, dim=1)
        model_prob = self.one_hot.repeat(target.size(0), 1).cuda()

        if len(target.shape) == 2: # if use onehot label, like mixup label
            target = torch.argmax(target, dim=1)                           # if target = [[0 1 0 0 0], [0 0 0 1 0], ...]  clw note: one-hot to class_id
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)

        #model_prob.scatter_(1, target.unsqueeze(1), self.confidence)      # if target = [1, 3, 4, 2, 0.... 0]

        if self.ignore_index >= 0:
            model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)
        # print("model_prob:{}".format(model_prob))
        # print("log_output:{}".format(log_output))

        return -torch.sum(model_prob * log_output) / target.size(0)


class LabelSmoothingLoss_clw(nn.Module):   # clw note: for mixup TODO
    def __init__(self, label_smoothing, class_nums, ignore_index=-100):
        assert 0.0 <= label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss_clw, self).__init__()

        self.label_smoothing = label_smoothing
        self.smoothing_value = label_smoothing / (class_nums - 1)


    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        log_output = F.log_softmax(output, dim=1)
        aaa = target == 1  #  no cutmix, and belongs to that class_id     # clw note: mixup, cutmix的样本,不做label smooth; 其他的正常做；这里把没有mixup的筛选出来
        bbb = aaa.sum(1) == 1
        ccc = ((target != 1) & (target != 0))  # cutmix, and belongs to that class_id
        ddd = ccc.sum(1) == 2

        target[bbb, :] = self.smoothing_value
        target[aaa] = 1 - self.label_smoothing
        target[ccc] -= 2.5 * self.smoothing_value     # - self.smoothing_value * 3 / 2 - self.smoothing_value
        target[ddd, :] += self.smoothing_value

        return -torch.sum(target * log_output) / target.size(0)