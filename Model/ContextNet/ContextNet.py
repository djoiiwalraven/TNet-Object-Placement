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

            # same out_channels??
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ContextNet(nn.Module):
    """
    U-Net architecture with attention gates in the skip connections.
    """
    def __init__(self, in_channels=1, out_channels=1, init_features=64, feature_length=4):
        super(ContextNet, self).__init__()

        self.features = [init_features * 2**i for i in range(feature_length)]

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        #self.attention_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Contracting path (encoder)
        channels = in_channels
        for feature in self.features:
            self.downs.append(ConvBlock(channels, feature))
            channels = feature

        # Tradditional Bottleneck
        self.bottleneck = ConvBlock(self.features[-1], self.features[-1]*2)

        #Qemb = None # Does it need pos.enc?
        #Kemb = Vemb = None # Does it need pos.enc? (probably not here)




        #Bottleneck A 
        #self.bottleneck_a = ConvBlock(self.features[-1], self.features[-1]*2)

        #Bottleneck Attention (bottleneck_a, 1d-tensor: context input)

        # Bottleneck B
        #self.bottleneck_b = #deconvolution


        # Expanding path (decoder)
        reversed_features = self.features[::-1]
        channels = self.features[-1]*2
        for feature in reversed_features:
            # Up-sampling using transpose convolution
            self.ups.append(
                nn.ConvTranspose2d(channels, feature, kernel_size=2, stride=2)
            )
            
            # Convolutional block after concatenation
            self.ups.append(ConvBlock(channels, feature))
            channels = feature

        # Final output layer
        self.final_conv = nn.Conv2d(self.features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        for down in self.downs:
            x = down(x)
            x = self.pool(x)

        # Bottleneck
        print(f'in bottle: {x.shape}')
        x = self.bottleneck(x)
        
        print(f'out bottle: {x.shape}')
        # Decoder path
        for idx in range(0, len(self.ups), 2):
            print()
            print('upsample: ')
            print(x.shape)
            # Up-sampling
            x = self.ups[idx](x)

            # Adjust dimensions if necessary
            #if x.shape != skip_connection.shape:
            #    x = TF.resize(x, size=skip_connection.shape[2:])

            #skip_connection = self.attention_blocks[idx // 2](g=x, x=skip_connection)

            # Concatenate
            #x = torch.cat((skip_connection, x), dim=1)

            # Convolutional block
            x = self.ups[idx + 1](x)

        # Final output layer with sigmoid activation for binary output
        return self.final_conv(x) # torch.sigmoid(self.final_conv(x))
