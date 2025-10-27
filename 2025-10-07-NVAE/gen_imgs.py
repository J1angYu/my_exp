import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import argparse
from pathlib import Path
from PIL import Image

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


def load_model(model_path, z_dim=512, device="cpu"):
    """加载预训练模型"""
    model = NVAE(z_dim=z_dim)
    model.apply(add_sn)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()
    return model


def generate_grid_images(model, img_size, output_dir, cols=12, rows=12, z_dim=512, device="cpu"):
    """生成网格图像"""
    width = cols * img_size
    height = rows * img_size
    result = np.zeros((width, height, 3), dtype=np.uint8)

    with torch.no_grad():
        z = torch.randn((cols * rows, z_dim, img_size//32, img_size//32)).to(device)
        gen_imgs, _ = model.decoder(z)
        gen_imgs = gen_imgs.reshape(rows, cols, 3, img_size, img_size)
        gen_imgs = gen_imgs.permute(0, 1, 3, 4, 2)
        gen_imgs = gen_imgs.cpu().numpy() * 255
        gen_imgs = gen_imgs.astype(np.uint8)

    for i in range(rows):
        for j in range(cols):
            result[i * img_size:(i + 1) * img_size, j * img_size:(j + 1) * img_size] = gen_imgs[i, j]

    # 保存网格图像
    im = Image.fromarray(result)
    grid_path = output_dir / "generated_grid.png"
    im.save(grid_path)
    print(f"生成的网格图像已保存到: {grid_path}")


def create_generation_comparison(model, dataset, output_dir, num_samples=8, z_dim=512, device="cpu"):
    """创建原图与生成图的对比"""
    # 随机选择样本用于对比
    dataset_size = len(dataset)
    sample_indices = random.sample(range(dataset_size), num_samples)
    print(f"对比样本索引: {sample_indices}")
    
    # 创建子图
    fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))
    
    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            # 获取原始图像
            img = dataset[idx].unsqueeze(0).to(device)
            ori_image = img.permute(0, 2, 3, 1)[0].cpu().numpy() * 255
            
            # 生成随机图像
            z = torch.randn((1, z_dim, 2, 2)).to(device)
            gen_imgs, _ = model.decoder(z)
            gen_image = gen_imgs.permute(0, 2, 3, 1)[0].cpu().numpy() * 255
            
            # 显示图像
            axes[0, i].imshow(ori_image.astype(np.uint8))
            axes[0, i].set_title(f"Real {idx}", fontsize=9)
            axes[0, i].axis('off')
            
            axes[1, i].imshow(gen_image.astype(np.uint8))
            axes[1, i].set_title(f"Generated", fontsize=9)
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    
    # 保存对比图
    comparison_path = output_dir / "generation_comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"生成对比图已保存到: {comparison_path}")


def main():
    parser = argparse.ArgumentParser(description='NVAE图像生成')
    parser.add_argument('--dataset', type=str, default='celeba', choices=['cifar10', 'celeba'],
                        help='数据集类型')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型文件路径')
    parser.add_argument('--mode', type=str, default='both', choices=['grid', 'comparison', 'both'],
                        help='生成模式: grid(网格图), comparison(对比图), both(两者)')
    parser.add_argument('--num_samples', type=int, default=8,
                        help='对比图样本数量')
    parser.add_argument('--grid_size', type=int, default=12,
                        help='网格大小')
    parser.add_argument('--device', type=str, default='cpu',
                        help='设备类型')
    parser.add_argument('--z_dim', type=int, default=512,
                        help='潜变量维度')
    
    args = parser.parse_args()
    
    # 获取数据集信息
    dataset_class, img_size, dataset_name = get_dataset_info(args.dataset)
    
    # 创建输出目录
    output_dir = Path(f"output/{dataset_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    model = load_model(args.model_path, z_dim=args.z_dim, device=args.device)
    
    # 生成网格图像
    if args.mode in ['grid', 'both']:
        generate_grid_images(model, img_size, output_dir, args.grid_size, args.grid_size, z_dim=args.z_dim, device=args.device)
    
    # 生成对比图像
    if args.mode in ['comparison', 'both']:
        # 加载数据集用于对比
        if args.dataset.lower() == 'cifar10':
            dataset = dataset_class(root='./datasets', train=True, download=True)
        else:  # celeba
            dataset = dataset_class(root='./datasets', img_dim=img_size, split='train')
        
        create_generation_comparison(model, dataset, output_dir, args.num_samples, z_dim=args.z_dim, device=args.device)


if __name__ == '__main__':
    main()
