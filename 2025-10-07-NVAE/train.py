import argparse
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from nvae.dataset import CIFAR10Dataset, CelebADataset     
from nvae.utils import add_sn
from nvae.vae import NVAE
from logger import Logger, setup_experiment_dir


class WarmupKLLoss:

    def __init__(self, init_weights, steps,
                 M_N=0.005,
                 eta_M_N=1e-5,
                 M_N_decay_step=3000):
        """
        预热KL损失，先对各级别的KL损失进行预热，预热完成后，对M_N的值进行衰减,所有衰减策略采用线性衰减
        :param init_weights: 各级别 KL 损失的初始权重
        :param steps: 各级别KL损失从初始权重增加到1所需的步数
        :param M_N: 初始M_N值
        :param eta_M_N: 最小M_N值
        :param M_N_decay_step: 从初始M_N值到最小M_N值所需的衰减步数
        """
        self.init_weights = init_weights
        self.M_N = M_N
        self.eta_M_N = eta_M_N
        self.M_N_decay_step = M_N_decay_step
        self.speeds = [(1. - w) / s for w, s in zip(init_weights, steps)]
        self.steps = np.cumsum(steps)
        self.stage = 0
        self._ready_start_step = 0
        self._ready_for_M_N = False
        self._M_N_decay_speed = (self.M_N - self.eta_M_N) / self.M_N_decay_step

    def _get_stage(self, step):
        while True:

            if self.stage > len(self.steps) - 1:
                break

            if step <= self.steps[self.stage]:
                return self.stage
            else:
                self.stage += 1

        return self.stage

    def get_loss(self, step, losses):
        loss = 0.
        stage = self._get_stage(step)

        for i, l in enumerate(losses):
            # Update weights
            if i == stage:
                speed = self.speeds[stage]
                t = step if stage == 0 else step - self.steps[stage - 1]
                w = min(self.init_weights[i] + speed * t, 1.)
            elif i < stage:
                w = 1.
            else:
                w = self.init_weights[i]

            # 如果所有级别的KL损失的预热都已完成
            if self._ready_for_M_N == False and i == len(losses) - 1 and w == 1.:
                # 准备M_N的衰减
                self._ready_for_M_N = True
                self._ready_start_step = step
            l = losses[i] * w
            loss += l

        if self._ready_for_M_N:
            M_N = max(self.M_N - self._M_N_decay_speed *
                      (step - self._ready_start_step), self.eta_M_N)
        else:
            M_N = self.M_N

        return M_N * loss


if __name__ == '__main__':
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    parser = argparse.ArgumentParser(description="Trainer for state AutoEncoder model.")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "celeba"], help="dataset to use")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="size of each sample batch")
    parser.add_argument("--z_dim", type=int, default=256, help="dimension of the latent space")
    parser.add_argument("--img_dim", type=int, default=32, help="dimension of the image")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--dataset_path", type=str, default="../data", help="dataset path ")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--exp_name", type=str, default="nvae", help="experiment name")
    opt = parser.parse_args()

    opt.exp_name += f"_{opt.dataset}"
    
    # 设置实验目录
    exp_dir = setup_experiment_dir(opt.exp_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 设置日志记录
    log_file = os.path.join(exp_dir, "training.log")
    
    epochs = opt.epochs
    batch_size = opt.batch_size
    dataset_path = opt.dataset_path

    # 根据数据集类型创建数据集
    if opt.dataset == "cifar10":
        train_ds = CIFAR10Dataset(root=dataset_path, train=True, download=True, img_dim=opt.img_dim)
    elif opt.dataset == "celeba":
        train_ds = CelebADataset(root=dataset_path, split='train', download=False, img_dim=opt.img_dim)
    else:
        raise ValueError(f"Unsupported dataset: {opt.dataset}")
    
    # 将整个数据集加载到GPU
    train_images = torch.stack([train_ds[i] for i in range(len(train_ds))]).to(device)
    train_ds = TensorDataset(train_images)
    
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    # 使用实验目录下的子目录
    checkpoints_dir = os.path.join(exp_dir, "checkpoints")
    output_dir = os.path.join(exp_dir, "output")

    # 创建模型
    model = NVAE(z_dim=opt.z_dim)

    # apply Spectral Normalization
    model.apply(add_sn)

    model.to(device)

    warmup_kl = WarmupKLLoss(init_weights=[1., 1. / 2, 1. / 8],
                             steps=[4500, 3000, 1500],
                             M_N=opt.batch_size / len(train_ds),
                             eta_M_N=5e-6,
                             M_N_decay_step=36000)

    optimizer = torch.optim.Adamax(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-4)

    # 使用Logger进行日志记录
    with Logger(log_file) as logger:
        print(f"开始训练 - 实验目录: {exp_dir}")
        print(f"数据集: {opt.dataset}")
        print(f"图像尺寸: {opt.img_dim}x{opt.img_dim}")
        print(f"设备: {device}")
        print(f"批次大小: {batch_size}")
        print(f"训练轮数: {epochs}")
        print(f"学习率: {opt.lr}")
        print(f"z_dim: {opt.z_dim}")
        print(f'M_N={warmup_kl.M_N}, ETA_M_N={warmup_kl.eta_M_N}')
        print(f"数据集大小: {len(train_ds)}")
        
        start_time = time.time()
        
        step = 0
        for epoch in range(epochs):
            model.train()

            for i, image in enumerate(train_dataloader):
                optimizer.zero_grad()

                image = image[0].to(device)
                image_recon, recon_loss, kl_losses = model(image)
                kl_loss = warmup_kl.get_loss(step, kl_losses)
                loss = recon_loss + kl_loss

                # 每100步打印一次日志
                if step % 100 == 0:
                    log_str = "---- [Epoch %d/%d, Step %d/%d] loss: %.6f, recon_loss: %.6f, kl_loss: %.6f ----" % (
                        epoch, epochs, i, len(train_dataloader), loss.item(), recon_loss.item(), kl_loss.item())
                    print(log_str)

                loss.backward()
                optimizer.step()

                step += 1

            # 生成并保存采样图
            model.eval()
            with torch.no_grad():
                z = torch.randn((1, opt.z_dim, 2, 2)).to(device)
                gen_img, _ = model.decoder(z)
                gen_img = gen_img.permute(0, 2, 3, 1)
                gen_img = gen_img[0].cpu().numpy() * 255
                gen_img = gen_img.astype(np.uint8)

                plt.imshow(gen_img)
                plt.savefig(os.path.join(output_dir, f"sample_epoch_{epoch}_loss_{loss.item():.6f}.png"))
                plt.close()
            model.train()

            scheduler.step()

            # 保存最后一个epoch的检查点
            checkpoint_path = os.path.join(checkpoints_dir, f"model_last_epoch.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"最后一个epoch检查点已保存: {checkpoint_path}")
        
        end_time = time.time()
        training_time = end_time - start_time
        print(f"训练完成! 总耗时: {training_time:.2f} 秒")
        print(f"所有文件保存在: {exp_dir}")
