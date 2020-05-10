import torch.nn as nn
import torch.nn.functional as F

def mean_squared(y, target, mask=None):
    y = y.softmax(1)
    loss = F.mse_loss(y, target, reduction="none").mean(1)
    if mask is not None:
        loss = mask * loss
    return loss.mean()

class MeanSquared(nn.Module):
    def forward(self, y, target, mask=None, *args, **kwargs):
        return mean_squared(y, target.detach(), mask)