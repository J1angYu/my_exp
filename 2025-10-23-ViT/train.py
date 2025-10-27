import os
import argparse
import logging
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from models.vit import ViT
from models.gqa_vit import GQAViT


def setup_distributed():
    """设置分布式训练环境"""

    if 'RANK' not in os.environ:
        return False, 0, 1, torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE")
        if key in os.environ
    }
    
    rank = int(env_dict['RANK'])
    local_rank = int(env_dict['LOCAL_RANK'])
    world_size = int(env_dict['WORLD_SIZE'])
    
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")

    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    backend = 'nccl'
    
    dist.init_process_group(backend=backend)
    
    print(
        f"[{os.getpid()}]: world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
    )
    
    return True, rank, world_size, device


def setup_logging(output_dir, rank=0):
    """设置日志配置"""
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, 'train.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=logging.WARNING)
    return logging.getLogger(__name__)


def get_model(model_name, in_channels, num_classes, **model_kwargs):
    """根据模型名称获取模型实例"""
    model_name = model_name.lower()
    
    if model_name == 'vit':
        return ViT(
            num_classes=num_classes,
            **model_kwargs
        )
    elif model_name == 'gqa_vit':
        return GQAViT(
            num_classes=num_classes,
            **model_kwargs
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}. Supported models: vit, gqa_vit")


def get_data_loaders(dataset_name, data_dir, batch_size, num_workers, image_size, distributed=False):
    """获取数据加载器和数据集信息"""
    configs = {
        'cifar10': (3, 10, [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010], datasets.CIFAR10),
        'cifar100': (3, 100, [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761], datasets.CIFAR100),
        'imagenet': (3, 1000, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], datasets.ImageFolder),
        'imagenet100': (3, 100, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], datasets.ImageFolder)
    }
    
    if dataset_name.lower() not in configs:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    in_channels, num_classes, mean, std, dataset_class = configs[dataset_name.lower()]
    
    if dataset_name.lower() in ['imagenet', 'imagenet100']:
        data_transform = {
            "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)]),
            "val": transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])}
        train_dataset = dataset_class(os.path.join(data_dir, 'train'), transform=data_transform["train"])
        val_dataset = dataset_class(os.path.join(data_dir, 'val'), transform=data_transform["val"])
    else:  # CIFAR-10/100
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        train_dataset = dataset_class(data_dir, train=True, download=True, transform=transform)
        val_dataset = dataset_class(data_dir, train=False, download=True, transform=transform)
    
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), 
                                   num_workers=num_workers, sampler=train_sampler, pin_memory=False)
    val_loader = Data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                                 num_workers=num_workers, sampler=val_sampler, pin_memory=False)
    
    return train_loader, val_loader, in_channels, num_classes


def train_epoch(model, train_loader, criterion, optimizer, device, logger, rank=0, clip_grad_norm=0.0):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc='Training', disable=(rank != 0))):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        if clip_grad_norm and clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        
        running_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device, rank=0, world_size=1):
    """验证模型"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc='Validating', disable=(rank != 0)):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    if world_size > 1:
        metrics = torch.tensor([val_loss, correct, total]).to(device) 
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        val_loss, correct, total = metrics[0].item(), metrics[1].item(), int(metrics[2].item())

    avg_loss = val_loss / (len(val_loader) * world_size) 
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def train_model(args, logger):
    """主训练函数"""
    is_distributed, rank, world_size, device = setup_distributed()
    
    if rank == 0:
        logger.info(f'Using device: {device}, World size: {world_size}, Distributed: {is_distributed}')
    
    # 获取数据加载器和数据集信息
    train_loader, val_loader, in_channels, num_classes = get_data_loaders(
        args.dataset, args.data_dir, args.batch_size, args.num_workers, 
        args.image_size, distributed=is_distributed
    )
    
    if rank == 0:
        logger.info(f'Dataset: {args.dataset}, Input channels: {in_channels}, Classes: {num_classes}')
    
    # 初始化模型
    model_kwargs = {
        'image_size': args.image_size,
        'patch_size': args.patch_size,
        'dim': args.dim,
        'depth': args.depth,
        'heads': args.heads,
        'mlp_dim': args.mlp_dim,
        'pool': args.pool,
        'channels': in_channels,
        'dropout': args.dropout,
        'emb_dropout': args.emb_dropout,
        'dim_head': args.dim_head,
    }
    
    if hasattr(args, 'num_kv_heads') and args.model == 'gqa_vit':
        model_kwargs['num_kv_heads'] = args.num_kv_heads
    
    model = get_model(args.model, in_channels, num_classes, **model_kwargs).to(device)

    if is_distributed:
        model = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None)
    
    if rank == 0:
        logger.info(f'Using model: {args.model}')
        if hasattr(args, 'num_kv_heads') and args.model == 'gqa_vit':
            logger.info(f'Using GQA with {args.num_kv_heads} kv heads')
    
    criterion = nn.CrossEntropyLoss()
    
    # 选择优化器
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    # use cosine scheduling
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    writer = None
    if rank == 0:
        writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard'))
        logger.info(f'Using TensorBoard for visualization')
        logger.info(f'Weight decay: {args.weight_decay}, Clip grad norm: {args.clip_grad_norm}')
    start_epoch = 0
    best_acc = 0.0
    if args.resume is not None:
        resume_path = args.resume if os.path.isfile(args.resume) else os.path.join(args.resume, 'checkpoint.pth')
        if os.path.exists(resume_path):
            ckpt = torch.load(resume_path, map_location=device)
            state = ckpt.get('model') or ckpt.get('state_dict')
            if state is not None:
                if is_distributed:
                    model.module.load_state_dict(state, strict=False)
                else:
                    model.load_state_dict(state, strict=False)
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            if 'scheduler' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_acc = ckpt.get('best_acc', 0.0)
            if rank == 0:
                logger.info(f'Resumed from {resume_path}: start_epoch={start_epoch}, best_acc={best_acc:.2f}%')
        else:
            if rank == 0:
                logger.warning(f'--resume specified but checkpoint not found: {resume_path}')
    # 训练记录
    best_model_state = None
    if rank == 0:
        logger.info(f'Starting training for {args.epochs} epochs')
    for epoch in range(start_epoch, args.epochs):
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        current_lr = optimizer.param_groups[0]['lr']
        if rank == 0:
            logger.info(f'Epoch {epoch+1}/{args.epochs} (lr={current_lr:.6f})')
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, logger, rank, clip_grad_norm=args.clip_grad_norm
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device, rank)
        scheduler.step()
        if rank == 0:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_acc, epoch)
            writer.add_scalar('Accuracy/Val', val_acc, epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)
            logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_state = model.module.state_dict() if is_distributed else model.state_dict()
                torch.save(best_model_state, os.path.join(args.output_dir, 'best_model.pth'))
                logger.info(f'New best model saved with accuracy: {best_acc:.2f}%')

            checkpoint = {
                'epoch': epoch,
                'model': model.module.state_dict() if is_distributed else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_acc': best_acc,
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'checkpoint_last.pth'))
    if rank == 0:
        writer.close()
        logger.info(f'Training completed. Best validation accuracy: {best_acc:.2f}%')
    if is_distributed:
        dist.destroy_process_group()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Vision Transformer Training Script')
    
    # 模型相关参数
    parser.add_argument('--model', type=str, default='vit', 
                       choices=['vit', 'gqa_vit'],
                       help='Model architecture (default: vit)')
    parser.add_argument('--image_size', type=int, default=32,
                       help='Image size (default: 32)')
    parser.add_argument('--patch_size', type=int, default=4,
                       help='Patch size for ViT (default: 4)')
    parser.add_argument('--dim', type=int, default=512,
                       help='Model dimension (default: 512)')
    parser.add_argument('--depth', type=int, default=6,
                       help='Number of transformer layers (default: 6)')
    parser.add_argument('--heads', type=int, default=6,
                       help='Number of attention heads (default: 6)')
    parser.add_argument('--num_kv_heads', type=int, default=3,
                       help='Number of key-value heads for GQA (default: 3)')
    parser.add_argument('--mlp_dim', type=int, default=512,
                       help='MLP dimension (default: 512)')
    parser.add_argument('--dropout', type=float, default=0.,
                       help='Dropout rate (default: 0.)')
    parser.add_argument('--emb_dropout', type=float, default=0.,
                       help='Embedding dropout rate (default: 0.)')
    parser.add_argument('--dim_head', type=int, default=64,
                       help='Attention head dimension (default: 64)')
    parser.add_argument('--pool', type=str, default='cls', choices=['cls', 'mean'],
                       help='Pooling strategy: cls or mean (default: cls)')
    
    # 数据相关参数
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'cifar100', 'imagenet', 'imagenet100'],
                       help='Dataset name (default: cifar10)')
    parser.add_argument('--data_dir', type=str, default='../data',
                       help='Data directory (default: ./data)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of workers for data loading (default: 8)')
    
    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer (default: adam)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay (default: 0.0). e.g., 0.3 for ImageNet.')
    parser.add_argument('--clip_grad_norm', type=float, default=0.0,
                       help='Global norm for gradient clipping (0 disables; e.g., 1.0 for ImageNet).')
    parser.add_argument('--output_dir', type=str, 
                       default=None,
                       help='Output directory for logs and checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint (file path or directory).')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        timestamp_ymd = datetime.now().strftime("%Y%m%d")
        timestamp_hms = datetime.now().strftime("%H%M%S")
        args.output_dir = f'./experiments/{timestamp_ymd}/{args.model}_{args.dataset}_{timestamp_hms}'
    
    return args


def main():
    """主函数"""
    args = parse_args()
    if args.resume is not None:
        if os.path.isdir(args.resume):
            args.output_dir = args.resume
        elif os.path.isfile(args.resume):
            resume_dir = os.path.dirname(args.resume)
            if resume_dir:
                args.output_dir = resume_dir
    rank = int(os.environ.get('RANK', 0))
    logger = setup_logging(args.output_dir, rank)
    if rank == 0:
        logger.info('Starting Vision Transformer training script')
        logger.info(f'Arguments: {vars(args)}')
        if 'WORLD_SIZE' in os.environ:
            logger.info(f'Distributed training detected: WORLD_SIZE={os.environ["WORLD_SIZE"]}, RANK={os.environ["RANK"]}')
    train_model(args, logger)


if __name__ == '__main__':
    main()