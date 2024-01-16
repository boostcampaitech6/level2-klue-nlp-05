import torch.nn.functional as F
import torch.nn as nn
import torch

class SmoothFocalCrossEntropyLoss(nn.Module):
  def __init__(self, smoothing=0.1, gamma=2.0):
    super(SmoothFocalCrossEntropyLoss, self).__init__()
    self.smoothing = smoothing
    self.gamma = gamma

  def forward(self, input, target):
    log_prob = F.log_softmax(input, dim=-1)
    prob = torch.exp(log_prob)
    weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.)
    weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
    
    # focal loss weight
    focal_weight = (1 - prob).pow(self.gamma)
    weight *= focal_weight

    loss = (-weight * log_prob).sum(dim=-1).mean()
    return loss