# Standard Library
from collections.abc import Iterable

# Third Party Library
import numpy as np
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, multiplier=1.0):
        self.multiplier = multiplier
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.cpu().detach().numpy()
        if isinstance(val, Iterable):
            val = np.array(val)
            self.update(np.mean(np.array(val)), n=val.size)
        else:
            self.val = self.multiplier * val
            self.sum += self.multiplier * val * n
            self.count += n
            self.avg = self.sum / self.count if self.count != 0 else 0

    def __str__(self) -> str:
        return f"{self.val:.6f} ({self.avg:.6f})"
