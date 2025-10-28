import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms

from utils import (
    BigVisionRandAugment, TwoHotMixUp, NativeScaler, adjust_learning_rate,
    misc
)
from plain_vit import create_plain_vit_small
from torch.utils.tensorboard import SummaryWriter


def get_args_parser():
    parser = argparse.ArgumentParser('ViT training', add_help=False)

    # Training parameters
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Per-GPU batch size (effective batch size is batch_size * num_gpus)')

    # Model parameters
    parser.add_argument('--model_name', default='vit_small_patch16_224', type=str,
                    help='Name of the ViT model to use from timm')
    parser.add_argument('--input_size', default=224, type=int,
                    help='Images input size for data augmentation (should match the model\'s expected input size)')
    parser.add_argument('--experiment_name', type=str, required=True,
                    help='Name of the experiment, used for saving checkpoints and logs')

    # Optimization parameters
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='base learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='epochs to warmup LR')
    parser.add_argument('--min_lr', type=float, default=0.0,
                        help='lower lr bound for cyclic schedulers')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='weight decay')

    # Augmentation parameters
    parser.add_argument('--randaug_n', type=int, default=2,
                        help='RandAugment number of ops')
    parser.add_argument('--randaug_m', type=int, default=10,
                        help='RandAugment magnitude')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                        help='mixup alpha value')

    # Dataset parameters
    parser.add_argument('--data_path', default='/path/to/imagenet',
                        help='dataset path')
    parser.add_argument('--output_dir', default=f'./experiments/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}',
                        help='path where to save checkpoints')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')


    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args))
    device = torch.device(args.device)
    set_seed(args)

    # Match data augmentation parameters with big_vision
    # - Use RandomResizedCrop with scale=(0.05, 1.0) matching big_vision's inception_crop
    # - Use our BigVisionRandAugment with the same parameters as big_vision
    # - Normalize to [-1, 1] range like big_vision's value_range(-1, 1)
    # - Match big_vision's resize_small(256) | central_crop(224)
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.05, 1.0), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        BigVisionRandAugment(num_ops=args.randaug_n, magnitude=args.randaug_m, fill=[128, 128, 128]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    mixup_fn = TwoHotMixUp(alpha=args.mixup_alpha) if args.mixup_alpha > 0 else None

    # Create datasets
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    dataset_val = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform_val)

    # Set up samplers for both training and validation
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True   # shuffle=True to reduce monitor bias, inherit from deit
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # Create dataloaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Create model
    model = create_plain_vit_small(model_name=args.model_name)
    model = model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_scaler = NativeScaler()

    print(f"Number of training images: {len(dataset_train)}")
    print(f"Number of validation images: {len(dataset_val)}")

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    writer = None
    if args.output_dir and misc.is_main_process():
        tb_dir = os.path.join(args.output_dir, "tensorboard")
        Path(tb_dir).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)

    for epoch in range(args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            mixup_fn, args=args,
        )
        test_stats = evaluate(data_loader_val, model, device)

        if writer is not None:
            writer.add_scalar('loss/train', train_stats.get('loss'), epoch)
            writer.add_scalar('loss/test', test_stats.get('loss'), epoch)
            writer.add_scalar('train/lr', train_stats.get('lr'), epoch)
            writer.add_scalar('test/acc1', test_stats.get('acc1'), epoch)
            writer.add_scalar('test/acc5', test_stats.get('acc5'), epoch)

        print(f"Accuracy of the model on {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
        }

        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        # Save checkpoint after each N epoch
        N = 5
        if (epoch + 1) % N == 0 or epoch == args.epochs - 1:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    if writer is not None:
        writer.close()


def train_one_epoch(model, data_loader,
                    optimizer, device, epoch, loss_scaler,
                    mixup_fn=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 20, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        lr = adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            # For MixUp with two hot targets
            samples, lamda, targets1, targets2 = mixup_fn(samples, targets)

            with torch.amp.autocast('cuda'):
                outputs = model(samples)
                loss = lamda * F.cross_entropy(outputs, targets1) + (1 - lamda) * F.cross_entropy(outputs, targets2)
        else:
            with torch.amp.autocast('cuda'):
                outputs = model(samples)
                loss = F.cross_entropy(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss_scaler(loss, optimizer, clip_grad=1.0,
                    parameters=model.parameters())

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=lr)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    model.eval()

    for batch in metric_logger.log_every(data_loader, 20, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            output = model(images)
            loss = F.cross_entropy(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def set_seed(args):
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = True


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
