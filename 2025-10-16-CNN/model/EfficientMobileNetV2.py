import torch
from torch import nn
from torchsummary import summary
import math


# 参考 EfficientNet 的通道与重复次数缩放
def round_filters(filters, width_mult=1.0, divisor=8, min_depth=None):
    if not width_mult:
        return int(filters)
    min_depth = min_depth or divisor
    filters *= width_mult
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, depth_mult=1.0):
    if not depth_mult:
        return int(repeats)
    return int(math.ceil(repeats * depth_mult))


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


class EfficientMobileNetV2(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, width_mult=1.0, depth_mult=1.0, divisor=8, min_depth=None):
        super().__init__()

        rf = lambda c: round_filters(c, width_mult=width_mult, divisor=divisor, min_depth=min_depth)
        rr = lambda r: round_repeats(r, depth_mult=depth_mult)

        ch32s = rf(32)
        ch16 = rf(16)
        ch24 = rf(24)
        ch32 = rf(32)
        ch64 = rf(64)
        ch96 = rf(96)
        ch160 = rf(160)
        ch320 = rf(320)
        ch1280 = rf(1280)

        r3 = rr(2)
        r4 = rr(3)
        r5 = rr(4)
        r6 = rr(3)
        r7 = rr(3)

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, ch32s, kernel_size=3, stride=1, padding=1, bias=False),
            NormAct6(ch32s),
        )

        self.block2 = LinearBottleNeck(ch32s, ch16, 1, 1)
        self.block3 = self._make_stage(r3, ch16, ch24, 2, 6)
        self.block4 = self._make_stage(r4, ch24, ch32, 2, 6)
        self.block5 = self._make_stage(r5, ch32, ch64, 2, 6)
        self.block6 = self._make_stage(r6, ch64, ch96, 1, 6)
        self.block7 = self._make_stage(r7, ch96, ch160, 1, 6)
        self.block8 = LinearBottleNeck(ch160, ch320, 1, 6)

        self.block9 = nn.Sequential(
            nn.Conv2d(ch320, ch1280, kernel_size=1, bias=False),
            NormAct6(ch1280),
        )

        self.block10 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch1280, num_classes, kernel_size=1),
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
    model = EfficientMobileNetV2(in_channels=3, num_classes=10, width_mult=1.4, depth_mult=1.4).to(device)
    print(summary(model, (3, 32, 32)))