import torch
from torch import nn
from Unet.ConvBlock import ConvBlock
import torchvision.transforms.functional as TF

class Unet(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 init_features = 64,
                 feature_length = 4,
                 ):
        super(Unet,self).__init__()
        self.features = [init_features * 2**i for i in range(feature_length)]

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in self.features:
            self.downs.append(ConvBlock(in_channels,feature))
            in_channels=feature

        for feature in reversed(self.features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2,feature,kernel_size=2, stride=2
                )
            )

            self.ups.append(ConvBlock(feature*2,feature))

        self.bottleneck = ConvBlock(self.features[-1], self.features[-1]*2)

        self.final_conv = nn.Conv2d(self.features[0], out_channels, kernel_size=1)


    def forward(self,x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = torch.sigmoid(self.bottleneck(x)) # Try adding sigmoid here

        skip_connections = skip_connections[::-1]

        for i, up in enumerate(self.ups):
            if i % 2 == 0:
                x = up(x)
            else:
                if x.shape != skip_connections[i//2].shape:
                    x = TF.resize(x, size=skip_connections[i//2].shape[2:])
                x = up(torch.cat((skip_connections[i//2],x),dim=1))


        return torch.sigmoid(self.final_conv(x))
    

def test():
    x = torch.randn((3,1,12,12))
    model = Unet(in_channels=1,out_channels=1,feature_length=2)
    preds = model(x)
    print(x.shape)
    print(preds.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()