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

class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception, self).__init__()
        
        #1x1conv branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=1, bias=False),
            NormAct(c1),
        )

        #1x1conv -> 3x3conv branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, c2[0], kernel_size=1, bias=False),
            NormAct(c2[0]),
            nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1, bias=False),
            NormAct(c2[1]),
        )

        #1x1conv -> 5x5conv branch
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, c3[0], kernel_size=1, bias=False),
            NormAct(c3[0]),
            nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2),
            NormAct(c3[1]),
        )

        #3x3maxpool -> 1x1conv branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, c4, kernel_size=1, bias=False),
            NormAct(c4),
        )

    def forward(self, x):
        o1 = self.branch1(x)
        o2 = self.branch2(x)
        o3 = self.branch3(x)
        o4 = self.branch4(x)
        return torch.cat((o1, o2, o3, o4), dim=1)
    

class GoogLeNet(nn.Module):
    def __init__(self, inception, in_channels=3, num_classes=10):
        super(GoogLeNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            NormAct(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            NormAct(64),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1, bias=False),
            NormAct(192),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block2 = nn.Sequential(
            inception(192, 64, (96, 128), (16, 32), 32),
            inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block3 = nn.Sequential(
            inception(480, 192, (96, 208), (16, 48), 64),
            inception(512, 160, (112, 224), (24, 64), 64),
            inception(512, 128, (128, 256), (24, 64), 64),
            inception(512, 112, (144, 288), (32, 64), 64),
            inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block4 = nn.Sequential(
            inception(832, 256, (160, 320), (32, 128), 128),
            inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout2d(p=0.4),
            nn.Flatten(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GoogLeNet(Inception, in_channels=3, num_classes=10).to(device)
    print(summary(model, (3, 32, 32)))