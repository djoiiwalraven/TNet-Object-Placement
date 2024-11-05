import numpy as np
import torch
from torch import nn


class VisTransformer(nn.Module):
    def __init__(self):
        super(VisTransformer,self).__init__()

    def forward(self,x):
        pass


A = torch.tensor([
        np.random.randn(32,4,4)
        .flatten()
    ]).to("cuda")

class Attention(nn.Module):
    def __init__(self,in_size=32,out_size=32):
        super(Attention,self).__init__()
        self.Q_weights = torch.tensor([np.random.randn(in_size,out_size)]).to("cuda")
        self.K_weights = torch.tensor([np.random.randn(in_size,out_size)]).to("cuda")
        self.V_weights = torch.tensor([np.random.randn(in_size,out_size)]).to("cuda")
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x: torch.Tensor):
        queries = x.matmul(self.Q_weights)
        keys = x.matmul(self.K_weights)
        values = x.matmul(self.V_weights)

        result: torch.Tensor = self.softmax(queries.mT.matmul(keys))
        return result.matmul(values.mT)
  

_,input_size = A.shape




att = Attention(in_size=input_size,out_size=input_size//2)

print(att(A))











