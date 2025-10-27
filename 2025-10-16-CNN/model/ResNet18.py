import torch
from torch import nn
from torchsummary import summary


class NormAct(nn.Module):
    """BN + ReLU"""
    def __init__(self, c):
        super().__init__()
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(x))

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=3, padding=1, stride=strides, bias=False),
            NormAct(num_channels),
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels)
        )
        self.act = nn.ReLU(inplace=True)
        
        if use_1x1conv:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=1, stride=strides, bias=False),
                nn.BatchNorm2d(num_channels)
            )
        else:
            self.shortcut = nn.Sequential()
            
    def forward(self, x):
        y = self.block(x)
        shortcut = self.shortcut(x)
        y = self.act(y + shortcut)
        return y

class ResNet18(nn.Module):
    def __init__(self, ResidualBlock, in_channels=3, num_classes=10):
        super(ResNet18, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            NormAct(64)
        )

        self.block2 = nn.Sequential(ResidualBlock(64, 64, use_1x1conv=False, strides=1),
                                    ResidualBlock(64, 64, use_1x1conv=False, strides=1))

        self.block3 = nn.Sequential(ResidualBlock(64, 128, use_1x1conv=True, strides=2),
                                    ResidualBlock(128, 128, use_1x1conv=False, strides=1))

        self.block4 = nn.Sequential(ResidualBlock(128, 256, use_1x1conv=True, strides=2),
                                    ResidualBlock(256, 256, use_1x1conv=False, strides=1))

        self.block5 = nn.Sequential(ResidualBlock(256, 512, use_1x1conv=True, strides=2),
                                    ResidualBlock(512, 512, use_1x1conv=False, strides=1))

        self.block6 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten(),
                                nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18(ResidualBlock, in_channels=3, num_classes=10).to(device)
    print(summary(model, (3, 32, 32)))