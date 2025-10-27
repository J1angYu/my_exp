import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def recon_bce(x, x_recon):
    """BCE重建损失"""
    return F.binary_cross_entropy(x_recon, x, reduction="sum")


# def recon_mse(x, x_recon):
#     """固定方差的高斯重建损失（MSE）：0.5 * ||x - x̂||^2"""
#     return 0.5 * torch.sum((x - x_recon) ** 2)

def recon_gaussian_nll(x, x_mu, x_logvar):
    """
    标准 Gaussian NLL（sum 归约，不含 log(2π) 常数项）
    0.5 * sum( (x-μ)^2 / σ^2 + log σ^2 )
    """
    inv_var = torch.exp(-x_logvar)
    diff2 = (x - x_mu) ** 2
    return 0.5 * torch.sum(diff2 * inv_var + x_logvar)

def kl_analytic(mu, logvar):
    """解析KL散度：KL(q(z|x)||p(z))，p(z)=N(0,I)"""
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return torch.sum(kl)


def kl_mc(mu, logvar, z=None):
    """蒙特卡洛KL散度估计"""
    # 计算对数密度
    log_q = -0.5 * torch.sum(
        math.log(2 * math.pi) + logvar + (z - mu).pow(2) / logvar.exp(), 
        dim=1
    )
    log_p = -0.5 * torch.sum(math.log(2 * math.pi) + z.pow(2), dim=1)
    kl_each = log_q - log_p
    return torch.sum(kl_each)


class VAELoss(nn.Module):
    """VAE损失函数，支持多种重建损失和KL散度计算方式"""
    
    def __init__(self, recon_type="bce", kl_type="analytic", beta=1.0):
        super().__init__()
        
        # 参数验证
        assert recon_type in ("bce", "gaussian"), f"不支持的重建损失类型: {recon_type}"
        assert kl_type in ("analytic", "mc"), f"不支持的KL类型: {kl_type}"

        self.recon_type = recon_type
        self.kl_type = kl_type
        self.beta = beta

    def forward(self, x, x_recon, mu, logvar, z=None, beta=None):
        """计算VAE总损失"""
        # 使用传入的beta值，如果没有则使用默认值
        current_beta = beta if beta is not None else self.beta
        
        # 重建损失
        if self.recon_type == "bce":
            rec_loss = recon_bce(x, x_recon)
        else:  # gaussian
            x_mu, x_logvar = x_recon
            rec_loss = recon_gaussian_nll(x, x_mu, x_logvar)

        # KL散度
        if self.kl_type == "analytic":
            kl_loss = kl_analytic(mu, logvar)
        else:
            kl_loss = kl_mc(mu, logvar, z)

        total_loss = rec_loss + current_beta * kl_loss
        return total_loss, rec_loss, kl_loss