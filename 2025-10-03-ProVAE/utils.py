import os
import sys
import json
from datetime import datetime
import torch
from torchvision import models


class Logger:
    """日志记录器，同时输出到终端和文件"""
    
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

    def __enter__(self):
        self._prev_stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.stdout = getattr(self, '_prev_stdout', self.terminal)
        self.close()


def setup_experiment(args):
    """设置实验目录和配置文件"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join("experiments", f"{args.exp_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(exp_dir, "config.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    print(f"实验目录: {exp_dir}")
    return exp_dir


def _preprocess_for_inception(x):
    """预处理图像用于InceptionV3特征提取"""
    if x.dtype != torch.float32:
        x = x.float()
    
    # 调整到299x299并转为3通道
    x = torch.nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    
    # ImageNet标准化
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def _get_inception_model(device):
    """获取InceptionV3模型和特征提取函数"""
    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    model.to(device).eval()

    def extract_features(imgs):
        features = []
        def hook_fn(module, input, output):
            features.append(output)
        
        handle = model.avgpool.register_forward_hook(hook_fn)
        with torch.no_grad():
            model(imgs)
        handle.remove()
        
        return torch.flatten(features[0], start_dim=1)

    return model, extract_features


def _compute_statistics(features):
    """计算特征的均值和协方差矩阵"""
    mu = features.mean(dim=0)
    centered = features - mu
    cov = (centered.T @ centered) / (features.shape[0] - 1)
    return mu, cov


def _matrix_sqrt(matrix):
    """计算对称矩阵的平方根"""
    eigenvals, eigenvecs = torch.linalg.eigh(matrix)
    eigenvals = torch.clamp(eigenvals, min=0)
    sqrt_eigenvals = torch.sqrt(eigenvals)
    return eigenvecs @ torch.diag(sqrt_eigenvals) @ eigenvecs.T


def _frechet_distance(mu1, cov1, mu2, cov2):
    """计算Frechet距离"""
    # 计算 (mu1 - mu2)^2
    diff = mu1 - mu2
    mean_diff = diff.dot(diff).item()
    
    # 计算 tr(cov1) + tr(cov2) - 2*tr(sqrt(cov1 @ cov2))
    cov1_sqrt = _matrix_sqrt(cov1)
    middle_term = _matrix_sqrt(cov1_sqrt @ cov2 @ cov1_sqrt)
    
    trace_sum = torch.trace(cov1).item() + torch.trace(cov2).item()
    trace_sqrt = 2.0 * torch.trace(middle_term).item()
    
    return mean_diff + trace_sum - trace_sqrt


def compute_fid_for_vae(vae_model, test_loader, device, input_dim, z_dim, image_shape=None):
    """计算VAE生成图像与真实图像的FID分数"""
    inception_model, extract_features = _get_inception_model(device)
    
    # 确定图像形状
    if image_shape is not None:
        height, width = image_shape
    else:
        side = int(input_dim ** 0.5)
        height, width = side, side

    real_features_list = []
    fake_features_list = []

    with torch.no_grad():
        for x_real, _ in test_loader:
            batch_size = x_real.size(0)
            
            # 处理真实图像
            if len(x_real.shape) == 4 and x_real.shape[1] == 1:
                x_real_processed = x_real
            elif len(x_real.shape) == 4:
                x_real_processed = x_real.mean(dim=1, keepdim=True)
            else:
                x_real_processed = x_real.view(-1, 1, height, width)
            
            # 提取真实图像特征
            x_real_prep = _preprocess_for_inception(x_real_processed.to(device))
            real_feat = extract_features(x_real_prep)
            real_features_list.append(real_feat.cpu())

            # 生成假图像
            z = torch.randn(batch_size, z_dim, device=device)
            
            if hasattr(vae_model, 'max_stage'):
                x_fake = vae_model.decode(z, stage=vae_model.max_stage, alpha=1.0)
            else:
                x_fake = vae_model.decode(z)
            
            # 处理gaussian模式的输出（可能是元组）
            if isinstance(x_fake, tuple):
                # gaussian模式：(mu, logvar)
                x_fake_mu, x_fake_logvar = x_fake
                # 使用均值作为生成图像，并限制到[0,1]范围
                x_fake = torch.clamp(x_fake_mu, 0, 1)
            
            # 处理生成图像
            if len(x_fake.shape) == 2:
                x_fake = x_fake.view(-1, 1, height, width)
            elif x_fake.shape[1] != 1:
                x_fake = x_fake.mean(dim=1, keepdim=True)
            
            # 提取生成图像特征
            x_fake_prep = _preprocess_for_inception(x_fake)
            fake_feat = extract_features(x_fake_prep)
            fake_features_list.append(fake_feat.cpu())

    # 计算统计量和FID
    real_features = torch.cat(real_features_list, dim=0)
    fake_features = torch.cat(fake_features_list, dim=0)
    
    mu_real, cov_real = _compute_statistics(real_features)
    mu_fake, cov_fake = _compute_statistics(fake_features)
    
    return _frechet_distance(mu_real, cov_real, mu_fake, cov_fake)