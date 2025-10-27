import torch.nn as nn
import torch


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self._seq = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, padding=2),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim), Swish(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            SELayer(dim))

    def forward(self, x):
        return x + 0.1 * self._seq(x)


class EncoderResidualBlock(nn.Module):
    """
    编码器侧的残差细化块：
      - 在不改变空间尺寸的前提下堆叠几层卷积，
        用于提炼局部特征、扩大感受野。
      - 末尾加上 SE 通道注意力；
        输出加回输入形成残差。
      - 乘上 0.1 的残差系数，防止训练初期梯度爆炸。

    结构：
      in :  B × C × H × W
        -> 5×5 Conv(stride=1,pad=2)      : B × C × H × W   # 扩大感受野
        -> 1×1 Conv                      : B × C × H × W   # 通道重混合
        -> BN + Swish
        -> 3×3 Conv(stride=1,pad=1)      : B × C × H × W
        -> SE Layer(C)                   : B × C × H × W
      out:  x + 0.1 × F(x)
      → B × C × H × W（尺寸不变）
    """
    def __init__(self, dim):
        super().__init__()

        self.seq = nn.Sequential(

            nn.Conv2d(dim, dim, kernel_size=5, padding=2),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim), Swish(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            SELayer(dim))

    def forward(self, x):
        return x + 0.1 * self.seq(x)


class DecoderResidualBlock(nn.Module):
    """
    生成器侧的残差细化块：
      - 相比编码器多了“分组卷积”n_group，
        用于降低计算量并模拟 depthwise 卷积。
      - 先 1×1 扩展 → 分组 5×5 卷积 → 1×1 压回，
        每步后接 BN + Swish + SE 注意力。
      - 残差缩放 0.1 保证数值稳定。

    结构：
      in :  B × C × H × W
        -> 1×1 Conv(expand)             : B × (n_group·C) × H × W
        -> BN + Swish
        -> 5×5 Grouped Conv(groups=n_group)
             : B × (n_group·C) × H × W   # 每组卷自己通道，降算力
        -> BN + Swish
        -> 1×1 Conv(project back)        : B × C × H × W
        -> BN + SE Layer(C)
      out:  x + 0.1 × F(x)
      → B × C × H × W（尺寸不变）
    """
    def __init__(self, dim, n_group):
        super().__init__()

        self._seq = nn.Sequential(
            nn.Conv2d(dim, n_group * dim, kernel_size=1),
            nn.BatchNorm2d(n_group * dim), Swish(),
            nn.Conv2d(n_group * dim, n_group * dim, kernel_size=5, padding=2, groups=n_group),
            nn.BatchNorm2d(n_group * dim), Swish(),
            nn.Conv2d(n_group * dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            SELayer(dim))

    def forward(self, x):
        return x + 0.1 * self._seq(x)
