import torch
from torch import nn
from torch import Tensor
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    def __init__(self,patch_size=3,emb_size = 64):
        self.patch_size = patch_size
        super().__init__()
        
        self.projection = nn.Sequential(
            Rearrange('b c (h ph) (w pw) -> b (h w) (ph pw c)', ph=patch_size,pw=patch_size),
            nn.Linear(self.patch_size * self.patch_size, emb_size)
        )

    def forward(self, x):
        x = self.projection(x)
        return x; 
