import torch
from torch.optim.lr_scheduler import _LRScheduler

class PolyLRScheduler(_LRScheduler):

    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, verbose=False):
        self.max_iters = max_iters
        self.power = power
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch >= self.max_iters:
            return [0.0 for base_lr in self.base_lrs]

        decay_factor = (1 - self.last_epoch / self.max_iters) ** self.power

        return [base_lr * decay_factor for base_lr in self.base_lrs]
