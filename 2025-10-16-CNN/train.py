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
from torch.optim.lr_scheduler import _LRScheduler

from model.ResNet18 import ResNet18, ResidualBlock
from model.VGG16 import VGG16
from model.GoogLeNet import GoogLeNet, Inception
from model.MobileNetV2 import MobileNetV2
from model.Xception import Xception, MiddleFlowBlock
from model.MobileNet import MobileNet
from model.EfficientMobileNetV2 import EfficientMobileNetV2 


def setup_logging(output_dir):
    """设置日志配置"""
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
    return logging.getLogger(__name__)


def get_dataset_info(dataset_name):
    """获取数据集信息"""
    dataset_configs = {
        'cifar10': {
            'in_channels': 3,
            'num_classes': 10,
            'normalize': transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            ),
            'dataset_class': datasets.CIFAR10
        },
        'cifar100': {
            'in_channels': 3,
            'num_classes': 100,
            'normalize': transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            ),
            'dataset_class': datasets.CIFAR100
        },
        'mnist': {
            'in_channels': 1,
            'num_classes': 10,
            'normalize': transforms.Normalize(
                mean=[0.1307],
                std=[0.3081]
            ),
            'dataset_class': datasets.MNIST
        }
    }
    
    if dataset_name.lower() not in dataset_configs:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: {list(dataset_configs.keys())}")
    
    return dataset_configs[dataset_name.lower()]


def get_model(model_name, in_channels, num_classes):
    """根据模型名称获取模型实例"""
    model_name = model_name.lower()
    
    if model_name == 'resnet18':
        return ResNet18(ResidualBlock, in_channels=in_channels, num_classes=num_classes)
    elif model_name == 'vgg16':
        return VGG16(in_channels=in_channels, num_classes=num_classes)
    elif model_name == 'googlenet':
        return GoogLeNet(Inception, in_channels=in_channels, num_classes=num_classes)
    elif model_name == 'mobilenet':
        return MobileNet(in_channels=in_channels, num_classes=num_classes)
    elif model_name == 'mobilenetv2':
        return MobileNetV2(in_channels=in_channels, num_classes=num_classes)
    elif model_name == 'efficientmobilenetv2':
        return EfficientMobileNetV2(in_channels=in_channels, num_classes=num_classes, width_mult=1.4, depth_mult=1.4)
    elif model_name == 'xception':
        return Xception(MiddleFlowBlock, in_channels=in_channels, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Supported models: resnet18, vgg16, googlenet, mobilenet, mobilenetv2, efficientmobilenetv2, xception")


def get_data_loaders(dataset_name, data_dir, batch_size, num_workers):
    """获取数据加载器"""
    dataset_config = get_dataset_info(dataset_name)
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        #transforms.Resize((32, 32)),  // MNIST 需要resize
        transforms.ToTensor(),
        dataset_config['normalize']
    ])
    
    val_transform = transforms.Compose([
        #transforms.Resize((32, 32)),  // MNIST 需要resize
        transforms.ToTensor(),
        dataset_config['normalize']
    ])
    
    train_dataset = dataset_config['dataset_class'](
        root=data_dir, train=True, download=True, transform=train_transform
    )
    val_dataset = dataset_config['dataset_class'](
        root=data_dir, train=False, download=True, transform=val_transform
    )
    
    train_loader = Data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = Data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device, logger, epoch, warmup_scheduler=None, warm_epochs=0):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc='Training')):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if warmup_scheduler is not None and epoch < warm_epochs:
            warmup_scheduler.step()
        
        running_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc='Validating'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    avg_loss = val_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def parse_lr_schedule(lr_schedule_str):
    """解析学习率调度字符串"""
    schedule = []
    for item in lr_schedule_str.split(','):
        epochs_str, lr_str = item.strip().split(':')
        epochs = int(epochs_str)
        lr = float(lr_str)
        schedule.append((epochs, lr))
    return schedule


def get_current_lr(epoch, lr_schedule):
    """根据当前epoch和学习率调度获取当前学习率"""
    current_epoch = 0
    for epochs, lr in lr_schedule:
        if epoch < current_epoch + epochs:
            return lr
        current_epoch += epochs
    # 如果超出了调度范围，返回最后一个学习率
    return lr_schedule[-1][1] if lr_schedule else 0.001

class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return [base_lr for base_lr in self.base_lrs]
        # 线性升温到 base_lr
        scale = float(self.last_epoch) / float(self.total_iters + 1e-8)
        return [base_lr * scale for base_lr in self.base_lrs]


def train_model(args, logger):
    """主训练函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # 获取数据集信息
    dataset_config = get_dataset_info(args.dataset)
    logger.info(f'Dataset: {args.dataset}, Input channels: {dataset_config["in_channels"]}, Classes: {dataset_config["num_classes"]}')
    
    # 获取数据加载器
    train_loader, val_loader = get_data_loaders(
        args.dataset, args.data_dir, args.batch_size, args.num_workers
    )
    
    # 初始化模型
    model = get_model(args.model, dataset_config['in_channels'], dataset_config['num_classes']).to(device)
    logger.info(f'Using model: {args.model}')
    
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    # 解析学习率调度
    lr_schedule = parse_lr_schedule(args.lr_schedule)
    logger.info(f'Learning rate schedule: {lr_schedule}')
    # Warmup 调度器
    iter_per_epoch = len(train_loader)
    warmup_scheduler = WarmUpLR(optimizer, total_iters=iter_per_epoch * args.warm) if args.warm > 0 else None

    # 设置TensorBoard
    writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard'))
    logger.info('Using TensorBoard for visualization')
    
    # 训练记录
    best_acc = 0.0
    best_model_state = None
    
    logger.info(f'Starting training for {args.epochs} epochs')
    
    for epoch in range(args.epochs):
        # 根据调度获取当前学习率
        current_lr = get_current_lr(epoch, lr_schedule)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        logger.info(f'Epoch {epoch+1}/{args.epochs} (lr={current_lr})')
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, logger, epoch, warmup_scheduler, args.warm
        )
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # TensorBoard记录
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)
        
        # 打印结果
        logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, os.path.join(args.output_dir, 'best_model.pth'))
            logger.info(f'New best model saved with accuracy: {best_acc:.2f}%')
    
    writer.close()
    logger.info(f'Training completed. Best validation accuracy: {best_acc:.2f}%')


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CNN Training Script')
    
    # 模型相关参数
    parser.add_argument('--model', type=str, default='resnet18', 
                       choices=['resnet18', 'vgg16', 'googlenet', 'mobilenet', 'mobilenetv2', 'efficientmobilenetv2', 'xception'],
                       help='Model architecture (default: resnet18)')
    
    # 数据相关参数
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'cifar100', 'mnist'],
                       help='Dataset name (default: cifar10)')
    parser.add_argument('--data_dir', type=str, default='../data',
                       help='Data directory (default: ../data)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of workers for data loading (default: 8)')
    
    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Learning rate (default: 0.1)')
    parser.add_argument('--lr_schedule', type=str, default='60:0.1,60:0.02,40:0.004,40:0.0008',
                       help='Learning rate schedule in format "epochs:lr,epochs:lr,..." (default: 60:0.1,60:0.02,40:0.004,40:0.0008)')
    parser.add_argument('--warm', type=int, default=1,
                       help='Warm-up epochs for per-batch scheduler (default: 1)')
    
    # 输出相关参数
    parser.add_argument('--output_dir', type=str, 
                       default=None,
                       help='Output directory for logs and checkpoints')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        timestamp_ymd = datetime.now().strftime("%Y%m%d")
        timestamp_hms = datetime.now().strftime("%H%M%S")
        args.output_dir = f'./experiments/{timestamp_ymd}/{args.model}_{args.dataset}_{timestamp_hms}'
    
    return args


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    logger = setup_logging(args.output_dir)
    logger.info('Starting CNN training script')
    logger.info(f'Arguments: {vars(args)}')
    
    # 开始训练
    train_model(args, logger)


if __name__ == '__main__':
    main()