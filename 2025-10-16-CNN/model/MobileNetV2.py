import torch
from torch import nn
from torchsummary import summary


class NormAct6(nn.Module):
    """BN + ReLU6"""
    def __init__(self, c):
        super().__init__()
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.ReLU6(inplace=True)
    def forward(self, x):
        return self.act(self.bn(x))


class LinearBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t=6):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        # expand -> BN+ReLU6
        self.expand_conv = nn.Conv2d(in_channels, in_channels * t, kernel_size=1, bias=False)
        self.expand_norm_act = NormAct6(in_channels * t)

        # depthwise -> BN+ReLU6
        self.depthwise = nn.Conv2d(in_channels * t, in_channels * t, kernel_size=3,
                                   stride=stride, padding=1, groups=in_channels * t, bias=False)
        self.depthwise_norm_act = NormAct6(in_channels * t)

        # pointwise -> BN
        self.pointwise = nn.Conv2d(in_channels * t, out_channels, kernel_size=1, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = self.expand_norm_act(self.expand_conv(x))
        y = self.depthwise_norm_act(self.depthwise(y))
        y = self.pointwise_bn(self.pointwise(y))
        if self.stride == 1 and self.in_channels == self.out_channels:
            y = y + x
        return y


class MobileNetV2(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32,  kernel_size=3, stride=1, padding=1),
            NormAct6(32),
        )

        self.block2 = LinearBottleNeck(32, 16, 1, 1)
        self.block3 = self._make_stage(2, 16, 24, 2, 6)
        self.block4 = self._make_stage(3, 24, 32, 2, 6)
        self.block5 = self._make_stage(4, 32, 64, 2, 6)
        self.block6 = self._make_stage(3, 64, 96, 1, 6)
        self.block7 = self._make_stage(3, 96, 160, 1, 6)
        self.block8 = LinearBottleNeck(160, 320, 1, 6)

        self.block9 = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            NormAct6(1280),
        )

        self.block10 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, num_classes, kernel_size=1),
            nn.Flatten()
        )

    def forward(self, x):
        for block in (self.block1, self.block2, self.block3, self.block4, self.block5,
                      self.block6, self.block7, self.block8, self.block9, self.block10):
            x = block(x)
        return x

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):
        layers = [LinearBottleNeck(in_channels, out_channels, stride, t)]
        layers += [LinearBottleNeck(out_channels, out_channels, 1, t) for _ in range(repeat - 1)]
        return nn.Sequential(*layers)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MobileNetV2(in_channels=3, num_classes=10).to(device)
    print(summary(model, (3, 32, 32)))