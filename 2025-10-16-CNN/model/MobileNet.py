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


class DepthSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3,
            stride=stride, padding=1, groups=in_channels, bias=False
        )
        self.dw_norm_act = NormAct(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pw_norm_act = NormAct(out_channels)

    def forward(self, x):
        x = self.dw_norm_act(self.depthwise(x))
        x = self.pw_norm_act(self.pointwise(x))
        return x


class MobileNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            NormAct(32)
        )

        self.features = nn.Sequential(
            DepthSeparableConv(32, 64, stride=1),
            DepthSeparableConv(64, 128, stride=2),
            DepthSeparableConv(128, 128, stride=1),
            DepthSeparableConv(128, 256, stride=2),
            DepthSeparableConv(256, 256, stride=1),
            DepthSeparableConv(256, 512, stride=2),
            DepthSeparableConv(512, 512, stride=1),
            DepthSeparableConv(512, 512, stride=1),
            DepthSeparableConv(512, 512, stride=1),
            DepthSeparableConv(512, 512, stride=1),
            DepthSeparableConv(512, 512, stride=1),
            DepthSeparableConv(512, 1024, stride=2),
            DepthSeparableConv(1024, 1024, stride=1),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MobileNet(in_channels=3, num_classes=10).to(device)
    print(summary(model, (3, 32, 32)))