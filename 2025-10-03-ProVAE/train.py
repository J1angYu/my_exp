import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from torchvision.utils import save_image

from utils import Logger, setup_experiment, compute_fid_for_vae
from models.ProVAE import ProVAE
from loss import VAELoss


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Progressive VAE Training')
    
    # 基本参数
    parser.add_argument('--exp_name', type=str, default='pro_vae', help='实验名称')
    parser.add_argument('--data_path', type=str, default='../data', help='数据路径')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'], help='数据集选择')
    parser.add_argument('--z_dim', type=int, default=20, help='潜在维度')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 损失函数参数
    parser.add_argument('--recon', type=str, default='bce', choices=['bce', 'gaussian'], help='重建损失类型')
    parser.add_argument('--kl', type=str, default='analytic', choices=['analytic', 'mc'], help='KL散度计算方式')
    parser.add_argument('--beta', type=float, default=1.0, help='KL散度权重')
    
    # KLD warm start参数
    parser.add_argument('--kld_warmup', action='store_true', help='启用KLD warm start')
    parser.add_argument('--kld_warmup_epochs', type=int, default=5, help='KLD warm start的epoch数')
    parser.add_argument('--kld_start_beta', type=float, default=0.0, help='KLD warm start的起始beta值')
    parser.add_argument('--kld_warmup_stages', type=int, nargs='+', default=[1], help='在哪些stage启用KLD warm start')

    # Progressive训练参数
    parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.999), help='Adam优化器betas参数')
    parser.add_argument('--epochs_per_stage', type=int, default=10, help='每阶段训练轮数')
    parser.add_argument('--fadein_ratio', type=float, default=0.5, help='渐入比例(0~1)')
    parser.add_argument('--start_res', type=int, default=4, help='起始分辨率')
    parser.add_argument('--final_res', type=int, default=32, help='最终分辨率')
    parser.add_argument('--base_ch', type=int, default=128, help='基础通道数')
    parser.add_argument('--min_ch', type=int, default=16, help='最小通道数')

    args = parser.parse_args()

    args.exp_name += f"_{args.dataset}_{args.recon}_{args.kl}"
    if args.kld_warmup:
        args.exp_name += f"_kldwarmup"
    
    return args


def get_dataset_config(dataset_name):
    """获取数据集配置"""
    if dataset_name == 'mnist':
        return {
            'channels': 1,
            'size': 28,
            'normalize': transforms.Normalize((0.5,), (0.5,))
        }
    elif dataset_name == 'cifar10':
        return {
            'channels': 3,
            'size': 32,
            'normalize': transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        }
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")


def create_data_loaders(args):
    """创建数据加载器"""
    os.makedirs(args.data_path, exist_ok=True)
    
    # 获取数据集配置
    config = get_dataset_config(args.dataset)
    
    # 数据预处理
    if args.dataset == 'mnist':
        # MNIST需要填充到目标分辨率
        padding = (args.final_res - config['size']) // 2
        transform = transforms.Compose([
            transforms.Pad(padding),
            transforms.ToTensor()
        ])
        
        train_data = MNIST(args.data_path, transform=transform, train=True, download=True)
        test_data = MNIST(args.data_path, transform=transform, train=False, download=True)
        
    elif args.dataset == 'cifar10':
        # CIFAR-10处理
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        train_data = CIFAR10(args.data_path, transform=transform, train=True, download=True)
        test_data = CIFAR10(args.data_path, transform=transform, train=False, download=True)
    
    loader_kwargs = {'num_workers': 4, 'pin_memory': True}
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    
    return train_loader, test_loader, config


def train_stage(model, train_loader, optimizer, criterion, device, stage, args):
    """训练单个阶段"""
    cur_res = model._res_of(stage)
    n_epochs = args.epochs_per_stage
    n_fade = int(n_epochs * args.fadein_ratio) if stage > 0 else 0
    
    print(f"\n===> Stage {stage} (res={cur_res}x{cur_res}) | fade-in epochs: {n_fade}/{n_epochs}")
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = recon_sum = kld_sum = 0.0
        
        # Alpha信息显示
        alpha_info = f"[alpha {epoch/n_fade:.2f}->{(epoch+1)/n_fade:.2f}]" if epoch < n_fade and n_fade > 0 else "[alpha 1.00]"
        
        # 在KLD warmup_stages 计算当前的KLD权重（warm start）
        if args.kld_warmup and stage in args.kld_warmup_stages:
            print(f"KLD warm start enabled in stage {stage}: {args.kld_start_beta:.3f} -> {args.beta:.3f} over {args.kld_warmup_epochs} epochs")
            if epoch < args.kld_warmup_epochs:
                # 线性增长从start_beta到target_beta
                current_beta = args.kld_start_beta + (args.beta - args.kld_start_beta) * (epoch + 1) / args.kld_warmup_epochs
            else:
                current_beta = args.beta
            beta_info = f"[beta {current_beta:.3f}]"
        else:
            current_beta = args.beta
            beta_info = f"[beta {current_beta:.3f}]"
        
        for i, (x, _) in enumerate(train_loader):
            x = x.to(device)
            x_tgt = F.interpolate(x, size=(cur_res, cur_res), mode="area")
            
            # 计算当前步骤的alpha值
            if stage > 0 and n_fade > 0 and epoch < n_fade:
                alpha = min(1.0, (epoch + 1) / n_fade)
            else:
                alpha = 1.0

            
            # 前向传播和反向传播
            optimizer.zero_grad()
            x_recon, mu, logvar, z = model(x, stage=stage, alpha=alpha)
            loss, recon_loss, kld_loss = criterion(x_tgt, x_recon, mu, logvar, z, beta=current_beta)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
            recon_sum += recon_loss.item()
            kld_sum += kld_loss.item()
        
        # 打印训练进度
        n_samples = len(train_loader.dataset)
        print(f"[Stage {stage}] Epoch {epoch+1}/{n_epochs} {alpha_info} {beta_info} - "
              f"Loss: {total_loss/n_samples:.4f}, Recon: {recon_sum/n_samples:.4f}, KLD: {kld_sum/n_samples:.4f}")


def save_stage_samples(model, device, stage, args, exp_dir):
    """保存当前阶段的生成样本"""
    model.eval()
    with torch.no_grad():
        z = torch.randn(64, args.z_dim, device=device)
        samples = model.decode(z, stage=stage, alpha=1.0)
        
        # 处理gaussian模式的输出
        if args.recon == "gaussian":
            samples_mu, samples_logvar = samples
            # 使用均值作为生成样本，并限制到[0,1]范围
            samples = torch.clamp(samples_mu, 0, 1)
        
        cur_res = model._res_of(stage)
        save_image(samples, os.path.join(exp_dir, f'sample_stage{stage}_{cur_res}x{cur_res}.png'), nrow=8)


def evaluate_model(model, test_loader, device, args, exp_dir):
    """评估模型并保存重建结果"""
    model.eval()
    with torch.no_grad():
        # 重建测试
        test_x, _ = next(iter(test_loader))
        test_x = test_x[:32].to(device)
        
        # 使用最终分辨率进行重建
        final_res = args.final_res
        x_tgt = F.interpolate(test_x, size=(final_res, final_res), mode="area")
        recon_x, _, _, _ = model(x_tgt, stage=model.max_stage, alpha=1.0)
        
        # 处理gaussian模式的输出
        if args.recon == "gaussian":
            recon_mu, recon_logvar = recon_x
            # 使用均值作为重建结果，并限制到[0,1]范围
            recon_x = torch.clamp(recon_mu, 0, 1)
        
        # 保存对比图像
        comparison = torch.cat([x_tgt[:16], recon_x[:16]])
        save_image(comparison, os.path.join(exp_dir, f'reconstruction_final_{final_res}.png'), nrow=8)
        
        # 计算FID
        fid = compute_fid_for_vae(model, test_loader, device, 
                                  input_dim=final_res*final_res, z_dim=args.z_dim)
        print(f"FID: {fid:.4f}")


def main(args):
    """主训练函数"""
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = setup_experiment(args)
    
    with Logger(os.path.join(exp_dir, 'training.log')):
        # 数据加载
        train_loader, test_loader, dataset_config = create_data_loaders(args)
        
        # 模型初始化
        model = ProVAE(
            in_ch=dataset_config['channels'],
            z_dim=args.z_dim,
            start_res=args.start_res, 
            final_res=args.final_res,
            base_ch=args.base_ch, 
            min_ch=args.min_ch,
            recon=args.recon,
        ).to(device)
        
        # 损失函数初始化
        criterion = VAELoss(
            recon_type=args.recon,
            kl_type=args.kl,
            beta=args.beta
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=args.betas)
        
        print(f"开始训练Progressive VAE ({args.dataset.upper()})...")
        
        # 逐阶段训练
        for stage in range(model.max_stage + 1):
            train_stage(model, train_loader, optimizer, criterion, device, stage, args)
            save_stage_samples(model, device, stage, args, exp_dir)
        
        print("训练完成!")
        
        # 保存模型和评估
        torch.save(model.state_dict(), os.path.join(exp_dir, 'pro_vae.pth'))
        evaluate_model(model, test_loader, device, args, exp_dir)
        print(f"结果已保存到 {exp_dir}")


if __name__ == '__main__':
    main(parse_args())
