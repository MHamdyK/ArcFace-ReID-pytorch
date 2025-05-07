import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcMarginProduct(nn.Module):
    """
    Implements ArcFace: Additive Angular Margin Loss.
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, labels):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - cosine.pow(2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = (one_hot * phi + (1.0 - one_hot) * cosine) * self.s
        return logits
