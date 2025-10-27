import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
import os
from pathlib import Path

from nvae.dataset import CIFAR10Dataset, CelebADataset
from nvae.utils import add_sn
from nvae.vae import NVAE


def get_dataset_info(dataset_name):
    """根据数据集名称返回数据集类和相关信息"""
    if dataset_name.lower() == 'cifar10':
        return CIFAR10Dataset, 32, 'cifar10'
    elif dataset_name.lower() == 'celeba':
        return CelebADataset, 64, 'celeba'
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")


def load_model(model_path, z_dim=256, device="cpu"):
    """加载预训练模型"""
    model = NVAE(z_dim=z_dim)
    model.apply(add_sn)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()
    return model


def create_reconstruction_comparison(model, dataset, output_dir, num_samples=8, device="cpu"):
    """创建重建对比图"""
    # 随机选择样本
    dataset_size = len(dataset)
    sample_indices = random.sample(range(dataset_size), num_samples)
    print(f"测试样本索引: {sample_indices}")
    
    # 创建子图
    fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))
    
    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            # 获取原始图像
            img = dataset[idx].unsqueeze(0).to(device)
            ori_image = img.permute(0, 2, 3, 1)[0].cpu().numpy() * 255
            
            # 重建图像
            gen_imgs, _, _ = model(img)
            recon_image = gen_imgs.permute(0, 2, 3, 1)[0].cpu().numpy() * 255
            
            # 显示图像
            axes[0, i].imshow(ori_image.astype(np.uint8))
            axes[0, i].set_title(f"Original {idx}", fontsize=9)
            axes[0, i].axis('off')
            
            axes[1, i].imshow(recon_image.astype(np.uint8))
            axes[1, i].set_title(f"Recon {idx}", fontsize=9)
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    
    # 保存图像
    output_path = output_dir / "reconstruction_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"重建对比图已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='NVAE重建测试')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'celeba'],
                        help='数据集类型')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型文件路径')
    parser.add_argument('--num_samples', type=int, default=8,
                        help='测试样本数量')
    parser.add_argument('--device', type=str, default='cpu',
                        help='设备类型')
    parser.add_argument('--z_dim', type=int, default=256,
                        help='潜变量维度')
    
    args = parser.parse_args()
    
    # 获取数据集信息
    dataset_class, img_size, dataset_name = get_dataset_info(args.dataset)
    
    # 创建输出目录
    output_dir = Path(f"output/{dataset_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据集
    if args.dataset.lower() == 'cifar10':
        dataset = dataset_class(root='../data', train=True, download=True)
    else:  # celeba
        dataset = dataset_class(root='../data', img_dim=img_size, split='train')
    
    # 加载模型
    model = load_model(args.model_path, z_dim=args.z_dim, device=args.device)
    
    # 创建重建对比图
    create_reconstruction_comparison(model, dataset, output_dir, args.num_samples, args.device)


if __name__ == '__main__':
    main()
