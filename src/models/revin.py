import torch
import torch.nn as nn

class RevIN(nn.Module):
    """
    Reversible instance normalization for univariate windows (Kim et al., 2022).
    normalize() z-scores each input window by its own mean/std; denormalize() maps outputs back.
    """
    def __init__(self, affine=True, eps=1e-5):
        super().__init__()
        self.affine = affine
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(1))
            self.bias = nn.Parameter(torch.zeros(1))

    def normalize(self, x):
        # x: (batch, input_len)
        self.mean = x.mean(dim=1, keepdim=True).detach()
        self.std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        x = (x - self.mean) / self.std
        if self.affine:
            x = x * self.weight + self.bias
        return x

    def denormalize(self, y):
        # y: (batch, horizon)
        if self.affine:
            y = (y - self.bias) / (self.weight + self.eps)
        return y * self.std + self.mean
