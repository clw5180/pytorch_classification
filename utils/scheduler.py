# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import math
from bisect import bisect_right
from typing import List

import torch
from torch.optim.lr_scheduler import _LRScheduler

__all__ = ["WarmupMultiStepLR", "WarmupCosineAnnealingLR"]


class WarmupMultiStepLR(_LRScheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            milestones: List[int],
            gamma: float = 0.1,
            warmup_factor: float = 0.001,
            warmup_iters: int = 1000,
            warmup_method: str = "linear",
            last_epoch: int = -1,
            **kwargs,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}", milestones
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        return [
            base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()

# https://github.com/JDAI-CV/fast-reid  新版本已经去掉了这个函数，改为直接用pytorch的 CosineAnnealing
class WarmupCosineAnnealingLR(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:
    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))
    When last_epoch=-1, sets initial lr as lr.
    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            max_iters: int,
            delay_iters: int = 0,
            eta_min_lr: int = 0,
            warmup_factor: float = 0.001,
            warmup_iters: int = 1000,
            warmup_method: str = "linear",
            last_epoch=-1,
            **kwargs
    ):
        self.max_iters = max_iters
        self.delay_iters = delay_iters
        self.eta_min_lr = eta_min_lr
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        assert self.delay_iters >= self.warmup_iters, "Scheduler delay iters must be larger than warmup iters"
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch <= self.warmup_iters:
            warmup_factor = _get_warmup_factor_at_iter(
                self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor,
            )
            return [
                base_lr * warmup_factor for base_lr in self.base_lrs
            ]
        elif self.last_epoch <= self.delay_iters:
            return self.base_lrs

        else:
            return [
                self.eta_min_lr + (base_lr - self.eta_min_lr) *
                (1 + math.cos(
                    math.pi * (self.last_epoch - self.delay_iters) / (self.max_iters - self.delay_iters))) / 2
                for base_lr in self.base_lrs]


def _get_warmup_factor_at_iter(
        method: str, iter: int, warmup_iters: int, warmup_factor: float
) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See https://arxiv.org/abs/1706.02677 for more details.
    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).
    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))

# https://github.com/Kylin9511/CRNet/blob/45427797c702bd88d7667ab703e9774900340676/utils/scheduler.py
class WarmUpCosineAnnealingLR2(_LRScheduler):
    def __init__(self, optimizer, T_max, T_warmup, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.eta_min = eta_min
        super(WarmUpCosineAnnealingLR2, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            return [base_lr * self.last_epoch / self.T_warmup for base_lr in self.base_lrs]
        else:
            k = 1 + math.cos(math.pi * (self.last_epoch - self.T_warmup) / (self.T_max - self.T_warmup))
            return [self.eta_min + (base_lr - self.eta_min) * k / 2 for base_lr in self.base_lrs]

# https://github.com/facebookresearch/meshrcnn/tree/ab0762f87bfd4cbdf4560982c5c33832a445380b
class WarmupCosineLR3(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        total_iters,
        warmup_iters=500,
        warmup_factor=0.1,  # 比如base_lr是0.05，这里就会从0.005开始往上升
        eta_min=0.0,
        last_epoch=-1,
        warmup_method="cosine",
    ):
        self.total_iters = total_iters
        self.warmup_iters = warmup_iters
        self.warmup_factor = warmup_factor
        assert warmup_method in ["linear", "cosine"]
        self.warmup_method = warmup_method
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                lr_factor = self.warmup_factor * (1 - alpha) + alpha
            elif self.warmup_method == "cosine":
                t = 1.0 + self.last_epoch / self.warmup_iters
                cos_factor = (1.0 + math.cos(math.pi * t)) / 2.0
                lr_factor = self.warmup_factor + (1.0 - self.warmup_factor) * cos_factor
            else:
                raise ValueError("Unsupported warmup method")
            return [lr_factor * base_lr for base_lr in self.base_lrs]

        num_decay_iters = self.total_iters - self.warmup_iters
        t = (self.last_epoch - self.warmup_iters) / num_decay_iters
        cos_factor = (1.0 + math.cos(math.pi * t)) / 2.0
        lrs = []
        for base_lr in self.base_lrs:
            lr = self.eta_min + (base_lr - self.eta_min) * cos_factor
            lrs.append(lr)
        return lrs

# 参考:SIIM-ISIC-Melanoma-Classification-1st-Place-Solution-master
# clw note: may have some bug, can be tested by scheduler.step() in while and print lr
from warmup_scheduler import GradualWarmupScheduler  # !pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
# scheduler
class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
    def get_lr(self):
        if self.last_epoch > self.total_epoch:  # clw note: ended warm up
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]