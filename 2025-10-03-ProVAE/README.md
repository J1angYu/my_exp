# **Progressive VAE**: 渐进式训练的VAE，多分辨率训练

### 数据集支持
- **MNIST**
- **CIFAR-10**

### 损失函数
- **重建损失**:
  - Binary Cross Entropy (BCE)
  - Mean Squared Error (MSE)
  - Gaussian Negative Log Likelihood (NLL)
- **KL散度**:
  - Analytic
  - per-sample estimate (MC)

### 实验管理
- 自动实验目录创建和配置保存
- 训练日志记录
- 模型检查点保存
- 可视化结果生成 (重建图像、生成样本)
- FID评估指标

### 训练Progressive VAE
```bash
# Progressive VAE训练
python train_provae.py --epochs_per_stage 10 --start_res 4 --final_res 32
```