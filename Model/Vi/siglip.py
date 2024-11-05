from typing import Optional, Tuple
import torch
from torch import nn
import numpy as np

class SiglipVisionConfig:

    def __init__(
            self,
            hidden_size=768, # Size of embedding vector (output_size)
            intermediate_size=3072, # Size of linear layer
            num_hidden_layers=12,
            num_attention_heads=12,
            num_channels=3,
            image_size=512, #256?
            patch_size=16,
            layer_norm_eps=1e-6,
            attention_dropout=0.0,
            num_image_tokens: int = None,
            **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens
    

class VisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super(VisionEmbeddings,self).__init__()
        self.config = config
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        print(self.patch_size)

        self.num_patches = (self.image_size // self.patch_size) ** 2
        print(self.num_patches)
        self.embed_dim = self.num_patches

        self.patch_embedding = nn.Conv2d(
            in_channels = config.num_channels,
            out_channels = self.embed_dim,
            kernel_size = self.patch_size,
            stride = self.patch_size,
            padding='valid'
        )

        
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1,-1)),
            persistent=False
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _,_,height,width = pixel_values.shape #[Batch Size, Channels, Height, Width]
        print(pixel_values.shape)
        patch_embeds = self.patch_embedding(pixel_values) #[Batch Size, Embed_Dim, Num_Patches, Num_Patches]
        #embeddings = patch_embeds.flatten(2) #[Batch Size, Embed_Dim, Num_Patches]
        #embeddings = embeddings.transpose(1,2) #[Batch Size, Num Patches, Embed_Dim]
        #embeddings = embeddings + self.position_embedding(self.position_ids) #[Batch Size, Num Patches, Embed_Dim(sizex2)]
        return patch_embeds
        


class SigLipTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super(SigLipTransformer,self).__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = VisionEmbeddings(config) # Does the convolutions + positional embeddings
        self.encoder = SigLipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim,eps=config.layer_norm_eps)

    def forward(self,pixel_values: torch.Tensor) -> torch.Tensor:
        return self.post_layernorm(self.encoder(self.embeddings(x)))


class SigLip(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super(SigLip,self).__init__()
        self.config = config
        self.vision_model = SigLipTransformer(config)

    def forward(self,pixel_values) -> Tuple:
        return self.vision_model(pixel_values=pixel_values)



if __name__ == "__main__":
    A = np.random.randn(1,3,512,512)
    A = torch.tensor(A,dtype=torch.float32)

    config = SiglipVisionConfig()

    embedder = VisionEmbeddings(config)
    embeddings = embedder(A)
    print(embeddings.shape)
    


