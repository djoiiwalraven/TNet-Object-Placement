import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class ConvBlock(nn.Module):
    """
    A convolutional block that consists of two convolutional layers,
    each followed by batch normalization and ReLU activation.
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class AttentionBlock(nn.Module):
    """
    An attention gate module that filters the features propagated through the skip connections.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        g: Output from the previous decoder layer (the gating signal)
        x: Corresponding encoder feature map (the skip connection)
        """
        # Apply the gating signal linear transformation
        g1 = self.W_g(g)
        # Apply the skip connection linear transformation
        x1 = self.W_x(x)
        # Combine and apply ReLU activation
        psi = self.relu(g1 + x1)
        # Apply the final attention coefficients
        psi = self.psi(psi)
        # Multiply attention coefficients with the skip connection features
        out = x * psi
        return out

class Tnet(nn.Module):
    """
    U-Net architecture with attention gates in the skip connections.
    """
    def __init__(self, in_channels=1, out_channels=1, init_features=64, feature_length=4):
        super(Tnet, self).__init__()

        self.features = [init_features * 2**i for i in range(feature_length)]

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Contracting path (encoder)
        channels = in_channels
        for feature in self.features:
            self.downs.append(ConvBlock(channels, feature))
            channels = feature

        # Bottleneck
        self.bottleneck = ConvBlock(self.features[-1], self.features[-1]*2)

        # Expanding path (decoder)
        reversed_features = self.features[::-1]
        channels = self.features[-1]*2
        for feature in reversed_features:
            # Up-sampling using transpose convolution
            self.ups.append(
                nn.ConvTranspose2d(channels, feature, kernel_size=2, stride=2)
            )
            # Attention block
            self.attention_blocks.append(
                AttentionBlock(F_g=feature, F_l=feature, F_int=feature // 2)
            )
            # Convolutional block after concatenation
            self.ups.append(ConvBlock(channels, feature))
            channels = feature

        # Final output layer
        self.final_conv = nn.Conv2d(self.features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Reverse the skip connections for decoding
        skip_connections = skip_connections[::-1]

        # Decoder path
        for idx in range(0, len(self.ups), 2):
            # Up-sampling
            x = self.ups[idx](x)

            # Skip connection
            skip_connection = skip_connections[idx // 2]

            # Adjust dimensions if necessary
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            # Apply attention
            skip_connection = self.attention_blocks[idx // 2](g=x, x=skip_connection)

            # Concatenate
            x = torch.cat((skip_connection, x), dim=1)

            # Convolutional block
            x = self.ups[idx + 1](x)

        # Final output layer with sigmoid activation for binary output
        return self.final_conv(x) # torch.sigmoid(self.final_conv(x))
