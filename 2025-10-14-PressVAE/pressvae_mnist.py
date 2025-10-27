import os
import argparse
from typing import Dict, Tuple
from datetime import datetime
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


# -------------------------
# 工具函数
# -------------------------
def downsample_to(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """双线性下采样到指定尺寸"""
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

def upsample_to(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """双线性上采样到指定尺寸"""
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """重参数化 z = mu + std * eps（eps ~ N(0, I)）"""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps

def kl_normal_standard(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL(q||N(0, I))，逐元素求和。
    KLD = 0.5 * (mu^2 + exp(logvar) - 1 - logvar)
    """
    return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1.0 - logvar)

def bce_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """
    二值交叉熵损失（忽略常数项 0.5*log(2π)），逐元素求和。
    BCE = -[ x*log(x_hat) + (1-x)*log(1-x_hat) ]
    """
    return F.binary_cross_entropy(x_hat, x, reduction="sum")

def gaussian_nll(x: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    """
    对角高斯的负对数似然（忽略常数项 0.5*log(2π)），逐元素求和。
    NLL = 0.5 * [ ((x - mu)/sigma)^2 + 2*log_sigma ]
    """
    inv_var = torch.exp(-2.0 * log_sigma)
    return 0.5 * torch.sum((x - mu) ** 2 * inv_var + 2.0 * log_sigma)


# -------------------------
# 基础层
# -------------------------
class NormAct(nn.Module):
    """BN + SiLU"""
    def __init__(self, c):
        super().__init__()
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(x))


# -------------------------
# 4x4
# 输入：X4 (B,3,4,4)
# -------------------------
class VAE4(nn.Module):
    def __init__(self, in_ch=3, z_ch=64, base_ch=32):
        super().__init__()
        # Encoder: (B,3,4,4) -> (B,z_ch*2,1,1)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3), NormAct(base_ch),          # -> (B,32,2,2)
            nn.Conv2d(base_ch, base_ch*2, 2), NormAct(base_ch*2),    # -> (B,64,1,1)
            nn.Conv2d(base_ch*2, z_ch*2, 1), NormAct(z_ch*2)   # -> (B,128,1,1)  // Z4
        )
        self.e_mu = nn.Conv2d(z_ch*2, z_ch, 1)        # -> (B,64,1,1)
        self.e_logvar = nn.Conv2d(z_ch*2, z_ch, 1)    # -> (B,64,1,1)

        # Decoder: (B,z_ch,1,1) -> (B,3,4,4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_ch, base_ch*2, 1), NormAct(base_ch*2),  # -> (B,64,1,1)
            nn.ConvTranspose2d(base_ch*2, base_ch, 2), NormAct(base_ch),    # -> (B,32,2,2)
            nn.ConvTranspose2d(base_ch, in_ch*2, 3), NormAct(in_ch*2)             # -> (B,3,4,4)  // X'4
        )
        self.out_mu = nn.Conv2d(in_ch*2, in_ch, 3, padding=1)        # -> (B,3,4,4)
        self.out_logsig = nn.Conv2d(in_ch*2, in_ch, 3, padding=1)    # -> (B,3,4,4)

    def encode(self, x4):
        z = self.encoder(x4)
        return self.e_mu(z), self.e_logvar(z)

    def decode(self, z):
        h = self.decoder(z)
        # 提取条件特征 (ConvT2)
        with torch.no_grad():
            cond_feat = self.decoder[:4](z)
        return torch.sigmoid(self.out_mu(h)), self.out_logsig(h), cond_feat

    def forward(self, x4):
        z_mu, z_logvar = self.encode(x4)
        z = reparameterize(z_mu, z_logvar)
        x_mu, x_logsig, cond_feat = self.decode(z)
        return {
            "x_mu": x_mu, "x_logsig": x_logsig,
            "z_mu": z_mu, "z_logvar": z_logvar,
            "cond_feat": cond_feat  # (B,32,2,2) -> 给 8x8
        }


# -------------------------
# 8x8
# 输入：D8 (B,3,8,8)，条件：来自 4x4 的 cond_feat (B,32,2,2)
# 解码时拼接 z8 与 cond_feat
# -------------------------
class VAE8(nn.Module):
    def __init__(self, in_ch=3, z_ch=64, base_ch=32, cond_in_ch=32):
        super().__init__()
        # Encoder: (B,3,8,8) -> (B,z_ch*2,2,2)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 4, stride=2, padding=1), NormAct(base_ch),    # -> (B,32,4,4)
            nn.Conv2d(base_ch, base_ch*2, 3), NormAct(base_ch*2),   # -> (B,64,2,2)
            nn.Conv2d(base_ch*2, z_ch*2, 1), NormAct(z_ch*2)  # -> (B,128,2,2)  // Z8
        )
        self.e_mu = nn.Conv2d(z_ch*2, z_ch, 1)        # -> (B,64,2,2)
        self.e_logvar = nn.Conv2d(z_ch*2, z_ch, 1)    # -> (B,64,2,2)

        # Decoder: (B,z_ch+cond_in_ch,2,2) -> (B,6,8,8)
        dec_in_ch = z_ch + cond_in_ch
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dec_in_ch, base_ch*2, 1), NormAct(base_ch*2),                     # -> (B,64,2,2)
            nn.ConvTranspose2d(base_ch*2, base_ch, 3), NormAct(base_ch),                            # -> (B,32,4,4)
            nn.ConvTranspose2d(base_ch, in_ch*2, 4, stride=2, padding=1), NormAct(in_ch*2)    # -> (B,6,8,8)  // D'8
        )
        self.out_mu = nn.Conv2d(in_ch*2, in_ch, 3, padding=1)        # -> (B,3,8,8)
        self.out_logsig = nn.Conv2d(in_ch*2, in_ch, 3, padding=1)    # -> (B,3,8,8)

    def encode(self, x8):
        z = self.encoder(x8)
        return self.e_mu(z), self.e_logvar(z)

    def decode(self, z, cond_feat):
        dec_input = torch.cat([z, cond_feat], dim=1) # (B,z_ch+cond_in_ch,2,2)
        h = self.decoder(dec_input)
        # 提取条件特征 (ConvT2)
        with torch.no_grad():
            cond_feat = self.decoder[:4](dec_input)
        return torch.sigmoid(self.out_mu(h)), self.out_logsig(h), cond_feat

    def forward(self, x8, cond_feat):
        z_mu, z_logvar = self.encode(x8)
        z = reparameterize(z_mu, z_logvar)
        x_mu, x_logsig, cond_feat = self.decode(z, cond_feat)
        return {
            "x_mu": x_mu, "x_logsig": x_logsig,
            "z_mu": z_mu, "z_logvar": z_logvar,
            "cond_feat": cond_feat  # (B,32,4,4) -> 给 16x16
        }


# -------------------------
# 16x16
# 输入：D16 (B,3,16,16)，条件：来自 8x8 的 cond_feat (B,32,4,4)
# 解码时拼接 z16 与 cond_feat
# -------------------------
class VAE16(nn.Module):
    def __init__(self, in_ch=3, z_ch=128, base_ch=32, cond_in_ch=32):
        super().__init__()
        # Encoder: (B,3,16,16) -> (B,z_ch*2,4,4)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 4, stride=2, padding=1), NormAct(base_ch), # -> (B,32,8,8)
            nn.Conv2d(base_ch, base_ch*2, 1), NormAct(base_ch*2),                # -> (B,64,8,8)
            nn.Conv2d(base_ch*2, base_ch*4, 4, stride=2, padding=1), NormAct(base_ch*4),  # -> (B,128,4,4)
            nn.Conv2d(base_ch*4, z_ch*2, 1), NormAct(z_ch*2)              # -> (B,256,4,4)  // Z16
        )
        self.e_mu = nn.Conv2d(z_ch*2, z_ch, 1)        # -> (B,128,4,4)
        self.e_logvar = nn.Conv2d(z_ch*2, z_ch, 1)    # -> (B,128,4,4)

        # Decoder: (B,z_ch+cond_in_ch,4,4) -> (B,6,16,16)
        dec_in_ch = z_ch + cond_in_ch
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dec_in_ch, base_ch*4, 1), NormAct(base_ch*4),  # -> (B,128,4,4)
            nn.ConvTranspose2d(base_ch*4, base_ch*2, 4, stride=2, padding=1), NormAct(base_ch*2),  # -> (B,64,8,8)
            nn.ConvTranspose2d(base_ch*2, base_ch, 1), NormAct(base_ch),         # -> (B,32,8,8)
            nn.ConvTranspose2d(base_ch, in_ch*2, 4, stride=2, padding=1), NormAct(in_ch*2)    # -> (B,6,16,16)  // D'16
        )
        self.out_mu = nn.Conv2d(in_ch*2, in_ch, 3, padding=1)        # -> (B,3,16,16)
        self.out_logsig = nn.Conv2d(in_ch*2, in_ch, 3, padding=1)    # -> (B,3,16,16)

    def encode(self, x16):
        z = self.encoder(x16)
        return self.e_mu(z), self.e_logvar(z)

    def decode(self, z, cond_feat):
        dec_input = torch.cat([z, cond_feat], dim=1) # (B,z_ch+cond_in_ch,4,4)
        h = self.decoder(dec_input)
        # 提取条件特征 (ConvT2)
        with torch.no_grad():
            cond_feat = self.decoder[:6](dec_input)
        return torch.sigmoid(self.out_mu(h)), self.out_logsig(h), cond_feat

    def forward(self, x16, cond_feat):
        z_mu, z_logvar = self.encode(x16)
        z = reparameterize(z_mu, z_logvar)
        x_mu, x_logsig, cond_feat = self.decode(z, cond_feat)
        return {
            "x_mu": x_mu, "x_logsig": x_logsig,
            "z_mu": z_mu, "z_logvar": z_logvar,
            "cond_feat": cond_feat  # (B,base_ch,8,8) -> 给 32x32
        }


# -------------------------
# 32x32
# 输入：D32 (B,3,32,32)，条件：来自 16x16 的 cond_feat (B,32,8,8)
# 解码时拼接 z32 与 cond_feat
# -------------------------
class VAE32(nn.Module):
    def __init__(self, in_ch=3, z_ch=128, base_ch=32, cond_in_ch=32):
        super().__init__()
        # Encoder: (B,3,32,32) -> (B,z_ch*2,8,8)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 4, stride=2, padding=1), NormAct(base_ch),           # -> (B,base_ch,16,16)
            nn.Conv2d(base_ch, base_ch*2, 1), NormAct(base_ch*2),    # -> (B,base_ch*2,16,16)
            nn.Conv2d(base_ch*2, base_ch*4, 4, stride=2, padding=1), NormAct(base_ch*4),  # -> (B,base_ch*4,8,8)
            nn.Conv2d(base_ch*4, z_ch*2, 1), NormAct(z_ch*2)   # -> (B,z_ch*2,8,8)  // Z32
        )
        self.e_mu = nn.Conv2d(z_ch*2, z_ch, 1)        # -> (B,z_ch,8,8)
        self.e_logvar = nn.Conv2d(z_ch*2, z_ch, 1)    # -> (B,z_ch,8,8)

        # Decoder: (B,z_ch+cond_in_ch,8,8) -> (B,6,32,32)
        dec_in_ch = z_ch + cond_in_ch
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dec_in_ch, base_ch*4, 1), NormAct(base_ch*4),  # -> (B,base_ch*4,8,8)
            nn.ConvTranspose2d(base_ch*4, base_ch*2, 4, stride=2, padding=1), NormAct(base_ch*2),    # -> (B,base_ch*2,16,16)
            nn.ConvTranspose2d(base_ch*2, base_ch, 1), NormAct(base_ch),         # -> (B,base_ch,16,16)
            nn.ConvTranspose2d(base_ch, in_ch*2, 4, stride=2, padding=1), NormAct(in_ch*2)     # -> (B,6,32,32)  // X'32
        )
        self.out_mu = nn.Conv2d(in_ch*2, in_ch, 3, padding=1)        # -> (B,3,32,32)
        self.out_logsig = nn.Conv2d(in_ch*2, in_ch, 3, padding=1)    # -> (B,3,32,32)

    def encode(self, x32):
        z = self.encoder(x32)
        return self.e_mu(z), self.e_logvar(z)

    def decode(self, z, cond_feat):
        dec_input = torch.cat([z, cond_feat], dim=1) # (B,z_ch+cond_in_ch,4,4)
        h = self.decoder(dec_input)
        # 提取条件特征 (ConvT2)
        with torch.no_grad():
            cond_feat = self.decoder[:6](dec_input)
        return torch.sigmoid(self.out_mu(h)), self.out_logsig(h), cond_feat

    def forward(self, x32, cond_feat):
        z_mu, z_logvar = self.encode(x32)
        z = reparameterize(z_mu, z_logvar)
        x_mu, x_logsig, cond_feat = self.decode(z, cond_feat)
        return {
            "x_mu": x_mu, "x_logsig": x_logsig,
            "z_mu": z_mu, "z_logvar": z_logvar,
            "cond_feat": cond_feat  # (B,base_ch,16,16)
        }


# -------------------------
# 多尺度容器
# -------------------------
class PressVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.vae4 = VAE4(in_ch=1, z_ch=64, base_ch=32)
        self.vae8 = VAE8(in_ch=1, z_ch=64, base_ch=32, cond_in_ch=32)
        self.vae16 = VAE16(in_ch=1, z_ch=128, base_ch=32, cond_in_ch=32)
        self.vae32 = VAE32(in_ch=1, z_ch=128, base_ch=32, cond_in_ch=32)

    @torch.no_grad()
    def reconstruct_pyramid(self, x32: torch.Tensor) -> Dict[str, torch.Tensor]:
        """给一个 32x32 图，多尺度逐级重建，返回各层重建的均值。"""
        x4  = downsample_to(x32, (4, 4))
        x8  = downsample_to(x32, (8, 8))
        x16 = downsample_to(x32, (16, 16))
        x32 = x32

        # 4x4
        o4 = self.vae4(x4)
        x4_mu = o4["x_mu"]
        f4 = o4["cond_feat"]

        # 8x8
        up4 = upsample_to(x4_mu, (8, 8))
        d8 = x8 - up4
        o8 = self.vae8(d8, f4)
        d8_mu = o8["x_mu"]
        x8_mu = up4 + d8_mu
        f8 = o8["cond_feat"]

        # 16x16
        up8 = upsample_to(x8_mu, (16, 16))
        d16 = x16 - up8
        o16 = self.vae16(d16, f8)
        d16_mu = o16["x_mu"]
        x16_mu = up8 + d16_mu
        f16 = o16["cond_feat"]

        # 32x32
        up16 = upsample_to(x16_mu, (32, 32))
        d32 = x32 - up16
        o32 = self.vae32(d32, f16)
        d32_mu = o32["x_mu"]
        x32_mu = up16 + d32_mu

        return {"x4": x4_mu, "x8": x8_mu, "x16": x16_mu, "x32": x32_mu}


# -------------------------
# 训练：按 stage 单独训练
# -------------------------
def make_pyramid_targets(x32: torch.Tensor):
    """构建金字塔真值：X4, X8, X16, X32"""
    X32 = x32
    X16 = downsample_to(x32, (16, 16))
    X8  = downsample_to(x32, (8, 8))
    X4  = downsample_to(x32, (4, 4))
    return X4, X8, X16, X32

def save_stage(model: PressVAE, out_dir: str, stage: int, logger=None):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"pressvae_stage{stage}.pt")
    torch.save(model.state_dict(), path)
    if logger:
        logger.info(f"[+] saved: {path}")
    else:
        print(f"[+] saved: {path}")

def load_until_stage(model: PressVAE, out_dir: str, stage: int, map_location=None, logger=None):
    """
    加载 <= stage 的权重（如果存在）。例如训练 stage=8 时会尝试加载 stage4 权重。
    """
    for s in (4, 8, 16, 32):
        if s > stage:
            break
        path = os.path.join(out_dir, f"pressvae_stage{s}.pt")
        if os.path.isfile(path):
            model.load_state_dict(torch.load(path, map_location=map_location, weights_only=True), strict=False)
            if logger:
                logger.info(f"[+] loaded: {path}")
            else:
                print(f"[+] loaded: {path}")

def freeze_below(model: PressVAE, stage: int):
    """冻结低于指定 stage 的子网参数。"""
    def freeze(m: nn.Module):
        for p in m.parameters():
            p.requires_grad_(False)

    if stage >= 8:
        freeze(model.vae4)
        model.vae4.eval()
    if stage >= 16:
        freeze(model.vae8)
        model.vae8.eval()
    if stage >= 32:
        freeze(model.vae16)
        model.vae16.eval()

def setup_logging(out_dir: str, stage: int):
    """设置日志记录，同时输出到终端和文件"""
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"stage{stage}_training.log")
    
    # 创建logger
    logger = logging.getLogger(f"pressvae_stage{stage}")
    logger.setLevel(logging.INFO)
    
    # 清除已有的handlers
    logger.handlers.clear()
    
    # 创建formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # 文件handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # 终端handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # 添加handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def train_stage(
    stage: int,
    dataset: str = "mnist",
    out_dir: str = "experiments/pressvae_mnist",
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    # 设置日志
    logger = setup_logging(out_dir, stage)
    logger.info(f"数据集: {dataset}")
    logger.info(f"开始训练 Stage {stage}")
    logger.info(f"参数设置: epochs={epochs}, batch_size={batch_size}, lr={lr}, device={device}")
    logger.info(f"输出目录: {out_dir}")

    # 数据集（MNIST/CIFAR-10）
    if dataset == "cifar10":
        transform = T.Compose([
            T.ToTensor(),  # [0,1]
            T.Normalize(mean=[0,0,0], std=[1,1,1])
        ])
        trainset = torchvision.datasets.CIFAR10(root="../data", train=True, download=True, transform=transform)
    elif dataset == "mnist":
        transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),  # [0,1]
            T.Normalize(mean=[0], std=[1])
        ])
        trainset = torchvision.datasets.MNIST(root="../data", train=True, download=True, transform=transform)
    else:
        raise ValueError("dataset must be in {'cifar10', 'mnist'}")
    loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    logger.info(f"数据集加载完成，训练样本数: {len(trainset)}")

    model = PressVAE().to(device)
    load_until_stage(model, out_dir, stage-1, map_location=device, logger=logger)  # 尝试加载前一阶段
    freeze_below(model, stage)

    # 只优化当前 stage 的参数
    if stage == 4:
        params = list(model.vae4.parameters())
    elif stage == 8:
        params = list(model.vae8.parameters())
    elif stage == 16:
        params = list(model.vae16.parameters())
    elif stage == 32:
        params = list(model.vae32.parameters())
    else:
        raise ValueError("stage must be in {4,8,16,32}")

    optimizer = torch.optim.Adam(params, lr=lr)

    # ---------- 可视化准备：固定一个可视化 batch ----------
    viz_dir = os.path.join(out_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)
    viz_n = min(16, batch_size)
    viz_loader = DataLoader(trainset, batch_size=viz_n, shuffle=False, num_workers=8, pin_memory=True)
    viz_x32, _ = next(iter(viz_loader))
    viz_x32 = viz_x32.to(device)

    # 辅助：根据当前阶段把重建结果提升到 32x32
    def reconstruct_at_stage(x32: torch.Tensor) -> torch.Tensor:
        X4, X8, X16, X32 = make_pyramid_targets(x32)
        model.eval()
        with torch.no_grad():
            if stage == 4:
                o4 = model.vae4(X4)
                recon = o4["x_mu"]
            elif stage == 8:
                o4 = model.vae4(X4)
                up4 = upsample_to(o4["x_mu"], (8, 8))
                o8 = model.vae8(X8 - up4, o4["cond_feat"])
                recon = up4 + o8["x_mu"]
            elif stage == 16:
                o4 = model.vae4(X4)
                up4 = upsample_to(o4["x_mu"], (8, 8))
                o8 = model.vae8(X8 - up4, o4["cond_feat"])
                x8_mu = up4 + o8["x_mu"]
                up8 = upsample_to(x8_mu, (16, 16))
                o16 = model.vae16(X16 - up8, o8["cond_feat"])
                recon = up8 + o16["x_mu"]
            else:  # stage == 32
                o4 = model.vae4(X4)
                up4 = upsample_to(o4["x_mu"], (8, 8))
                o8 = model.vae8(X8 - up4, o4["cond_feat"])
                x8_mu = up4 + o8["x_mu"]
                up8 = upsample_to(x8_mu, (16, 16))
                o16 = model.vae16(X16 - up8, o8["cond_feat"])
                x16_mu = up8 + o16["x_mu"]
                up16 = upsample_to(x16_mu, (32, 32))
                o32 = model.vae32(X32 - up16, o16["cond_feat"])
                recon = up16 + o32["x_mu"]
        return torch.clamp(recon, 0.0, 1.0)

    # 辅助：从标准正态采样生成到 32x32（按阶段拼接条件）
    def generate_at_stage(bs: int) -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            if stage == 4:
                z4 = torch.randn(bs, 64, 1, 1, device=device)
                x4_mu, _, cond4 = model.vae4.decode(z4)
                gen = x4_mu
            elif stage == 8:
                z4 = torch.randn(bs, 64, 1, 1, device=device)
                x4_mu, _, cond4 = model.vae4.decode(z4)
                up4 = upsample_to(x4_mu, (8, 8))
                z8 = torch.randn(bs, 64, 2, 2, device=device)
                d8_mu, _, cond8 = model.vae8.decode(z8, cond4)
                gen = up4 + d8_mu
            elif stage == 16:
                z4 = torch.randn(bs, 64, 1, 1, device=device)
                x4_mu, _, cond4 = model.vae4.decode(z4)
                up4 = upsample_to(x4_mu, (8, 8))
                z8 = torch.randn(bs, 64, 2, 2, device=device)
                d8_mu, _, cond8 = model.vae8.decode(z8, cond4)
                x8_mu = up4 + d8_mu
                up8 = upsample_to(x8_mu, (16, 16))
                z16 = torch.randn(bs, 128, 4, 4, device=device)
                d16_mu, _, cond16 = model.vae16.decode(z16, cond8)
                gen = up8 + d16_mu
            else:  # stage == 32
                z4 = torch.randn(bs, 64, 1, 1, device=device)
                x4_mu, _, cond4 = model.vae4.decode(z4)
                up4 = upsample_to(x4_mu, (8, 8))
                z8 = torch.randn(bs, 64, 2, 2, device=device)
                d8_mu, _, cond8 = model.vae8.decode(z8, cond4)
                x8_mu = up4 + d8_mu
                up8 = upsample_to(x8_mu, (16, 16))
                z16 = torch.randn(bs, 128, 4, 4, device=device)
                d16_mu, _, cond16 = model.vae16.decode(z16, cond8)
                x16_mu = up8 + d16_mu
                up16 = upsample_to(x16_mu, (32, 32))
                z32 = torch.randn(bs, 128, 8, 8, device=device)
                d32_mu, _ = model.vae32.decode(z32, cond16)
                gen = up16 + d32_mu
        return torch.clamp(gen, 0.0, 1.0)

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = recon_sum = kl_sum = 0.0
        n_pix = 0
        beta = 10
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            X4, X8, X16, X32 = make_pyramid_targets(x)

            optimizer.zero_grad(set_to_none=True)

            if stage == 4:
                o4 = model.vae4(X4)
                recon = bce_loss(X4, o4["x_mu"])
                kl = kl_normal_standard(o4["z_mu"], o4["z_logvar"])
                loss = recon + beta * kl
            elif stage == 8:
                with torch.no_grad():
                    o4 = model.vae4(X4)
                    up4 = upsample_to(o4["x_mu"], (8, 8))
                    cond4 = o4["cond_feat"]
                D8 = X8 - up4
                o8 = model.vae8(D8, cond4)
                recon = bce_loss(D8, o8["x_mu"])
                kl = kl_normal_standard(o8["z_mu"], o8["z_logvar"])
                loss = recon + beta * kl

            elif stage == 16:
                with torch.no_grad():
                    o4 = model.vae4(X4)
                    up4 = upsample_to(o4["x_mu"], (8, 8))
                    o8 = model.vae8(X8 - up4, o4["cond_feat"])
                    up8 = upsample_to(up4 + o8["x_mu"], (16, 16))
                    cond8 = o8["cond_feat"]
                D16 = X16 - up8
                o16 = model.vae16(D16, cond8)
                recon = bce_loss(D16, o16["x_mu"])
                kl = kl_normal_standard(o16["z_mu"], o16["z_logvar"])
                loss = recon + beta * kl

            else:  # stage == 32
                with torch.no_grad():
                    o4 = model.vae4(X4)
                    up4 = upsample_to(o4["x_mu"], (8, 8))
                    o8 = model.vae8(X8 - up4, o4["cond_feat"])
                    x8_mu = up4 + o8["x_mu"]
                    up8 = upsample_to(x8_mu, (16, 16))
                    o16 = model.vae16(X16 - up8, o8["cond_feat"])
                    x16_mu = up8 + o16["x_mu"]
                    cond16 = o16["cond_feat"]
                    up16 = upsample_to(x16_mu, (32, 32))
                D32 = X32 - up16
                o32 = model.vae32(D32, cond16)
                recon = bce_loss(D32, o32["x_mu"])
                kl = kl_normal_standard(o32["z_mu"], o32["z_logvar"])
                loss = recon + beta * kl

            loss.backward()
            optimizer.step()

            # 统计（以像素数归一化便于对比）
            b = x.size(0)
            if stage == 4:
                pix = b * 4 * 4 * 1
            elif stage == 8:
                pix = b * 8 * 8 * 1
            elif stage == 16:
                pix = b * 16 * 16 * 1
            else:
                pix = b * 32 * 32 * 1

            loss_sum += loss.item()
            recon_sum += recon.item()
            kl_sum += kl.item()
            n_pix += pix

        logger.info(f"[Stage {stage}] Epoch {epoch:03d}  "
                    f"loss/pix={loss_sum/n_pix:.6f}  recon/pix={recon_sum/n_pix:.6f}  kl/pix={kl_sum/n_pix:.6f}")

        # ---------- 可视化：保存原始/重建/生成拼图 ----------
        if epoch % 5 == 0:
            with torch.no_grad():
                # 获取当前stage对应分辨率的原图
                X4, X8, X16, X32 = make_pyramid_targets(viz_x32)
                if stage == 4:
                    orig = torch.clamp(X4, 0.0, 1.0)
                elif stage == 8:
                    orig = torch.clamp(X8, 0.0, 1.0)
                elif stage == 16:
                    orig = torch.clamp(X16, 0.0, 1.0)
                else:  # stage == 32
                    orig = torch.clamp(X32, 0.0, 1.0)
                
                recon = reconstruct_at_stage(viz_x32)
                gen = generate_at_stage(viz_x32.size(0))

                imgs = torch.cat([orig, recon, gen], dim=0)  # 3 行：原始、重建、生成
                grid = torchvision.utils.make_grid(imgs.cpu(), nrow=viz_n)
                save_path = os.path.join(viz_dir, f"stage{stage}_epoch{epoch:03d}.png")
                torchvision.utils.save_image(grid, save_path)
                print(f"[viz] saved: {save_path}")

    save_stage(model, out_dir, stage, logger=logger)


# -------------------------
# 命令行入口
# -------------------------
def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = f"experiments/pressvae_mnist_{timestamp}"
    
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, choices=[4,8,16,32], required=True, help="训练的分辨率阶段")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out_dir", type=str, default=out_dir)
    p.add_argument("--dataset", type=str, choices=["cifar10", "mnist"], default="mnist")
    args = p.parse_args()

    train_stage(stage=args.stage, dataset=args.dataset, out_dir=args.out_dir, epochs=args.epochs,
                batch_size=args.batch_size, lr=args.lr)

if __name__ == "__main__":
    main()