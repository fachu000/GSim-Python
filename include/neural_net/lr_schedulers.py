import math
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineMinLRScheduler(_LRScheduler):
    """
    Learning rate scheduler that implements a linear warmup phase followed by a
    cosine decay to a minimum learning rate.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1

        lrs = []
        for base_lr in self.base_lrs:
            alpha = self.min_lr / base_lr

            if step <= self.warmup_steps:
                lr = base_lr * step / self.warmup_steps
            else:
                progress = (step - self.warmup_steps) / max(
                    1, self.total_steps - self.warmup_steps)
                progress = min(progress, 1.0)

                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr = base_lr * (alpha + (1.0 - alpha) * cosine)

            lrs.append(lr)

        return lrs
