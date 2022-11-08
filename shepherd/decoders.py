import torch
import torch.nn.functional as F


def bilinear(s, r, t):
    return torch.sum(s * r * t, dim = 1)

def trans(s, r, t):
    return -torch.norm(s + r - t, dim = 1)

def dot(s, t):
    return torch.sum(s * t, dim = 1)
