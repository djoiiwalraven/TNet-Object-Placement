import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class SelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma      = nn.Parameter(torch.zeros(1))
        self.softmax    = nn.Softmax(dim=-1)
    
    def forward(self, x):
        batch_size, C, width, height = x.size()
        # Flatten the spatial dimensions
        proj_query = self.query_conv(x).view(batch_size, self.num_heads, C // self.num_heads, -1)
        proj_query = proj_query.permute(0, 1, 3, 2)  # (B, num_heads, N, C_head)
        
        proj_key = self.key_conv(x).view(batch_size, self.num_heads, C // self.num_heads, -1)
        # (B, num_heads, C_head, N)
        
        energy = torch.matmul(proj_query, proj_key)  # (B, num_heads, N, N)
        attention = self.softmax(energy)
        
        proj_value = self.value_conv(x).view(batch_size, self.num_heads, C // self.num_heads, -1)
        proj_value = proj_value.permute(0, 1, 3, 2)  # (B, num_heads, N, C_head)
        
        out = torch.matmul(attention, proj_value)  # (B, num_heads, N, C_head)
        out = out.permute(0, 1, 3, 2).contiguous()
        out = out.view(batch_size, C, width, height)
        
        out = self.gamma * out + x
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=False, num_heads=8):
        super(ConvBlock, self).__init__()
        self.use_attention = use_attention
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        if self.use_attention:
            self.attention = SelfAttention(out_channels, num_heads)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        if self.use_attention:
            x = self.attention(x)
        return x

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        # g: gating signal from decoder
        # x: skip connection from encoder
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
       
        psi = self.relu(g1 * x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class Anet(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 init_features=64,
                 feature_length=4,
                 use_attention=False,
                 num_heads=8):
        super(Anet, self).__init__()
        self.features = [init_features * 2**i for i in range(feature_length)]
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.use_attention = use_attention

        # Encoder path
        for feature in self.features:
            self.downs.append(ConvBlock(in_channels, feature, use_attention=use_attention, num_heads=num_heads))
            in_channels = feature

        # Bottleneck
        self.bottleneck = ConvBlock(self.features[-1], self.features[-1]*2, use_attention=use_attention, num_heads=num_heads)

        # Decoder path
        reversed_features = list(reversed(self.features))
        for idx, feature in enumerate(reversed_features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(ConvBlock(feature*2, feature, use_attention=use_attention, num_heads=num_heads))
            # Attention Gate
            self.attention_gates.append(
                AttentionGate(F_g=feature, F_l=feature, F_int=feature//2)
            )

        self.final_conv = nn.Conv2d(self.features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            

        # Bottleneck
        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        # Decoder
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)  # Upsample
            # Apply attention gate

            skip_connection = skip_connections[i//2]

            skip_connection = self.attention_gates[i//2](x, skip_connection)
            # Concatenate
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i+1](x)  # ConvBlock

        return torch.sigmoid(self.final_conv(x))

def test():
    x = torch.randn((3, 1, 256, 256))
    model = Anet(in_channels=1, out_channels=1, feature_length=4, use_attention=True)
    preds = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {preds.shape}")
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()
