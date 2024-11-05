import numpy as np

import torch
from torch import nn

A = np.random.randn(8,8)
X = np.random.randn(4)
Xt = np.zeros(8,8)



C = np.matmul(A,X)

print(C)

class CrossAttention(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(CrossAttention, self).__init__()


    def forward(self,q,k):
        pass




