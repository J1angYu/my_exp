# Plain ViT-S/16 ImageNet-1K Pre-training in PyTorch

A clean, simple, efficient, and hackable PyTorch implementation of state-of-the-art Vision Transformer Small (ViT-S/16) pre-training on ImageNet-1K that achieves >76% top-1 accuracy after 90 epochs of training. This repository reproduces the results from ["Better plain ViT baselines for ImageNet-1k"](https://arxiv.org/abs/2205.01580) in PyTorch with some minor implementation differences.

## Quick Start

Train a ViT-S/16 model on ImageNet-1K with 8 GPUs:

```bash
torchrun --nproc_per_node=8 main.py --data_path ~/autodl-tmp/imagenet --model_name vit_small_patch16_224 --experiment_name vit_small_i1k --epochs 90 --batch_size 128 --lr 1e-3 --weight_decay 0.1
```

## Implementation Details

We made several key improvements compared to training a standard timm ViT:

- **Optimizer Configuration**: Set the correct weight_decay value (0.1) for AdamW
- **Data Augmentation**:
  - Implemented RandAug to match big_vision's approach (differs from both PyTorch and timm implementations)
  - Updated RandomResizedCrop to use scale=(0.05, 1.0) instead of scale=(0.08, 1.0)
  - Changed image normalization to [-1, 1] range
- **MixUp Implementation**: Implemented TwoHotMixUp to match big_vision's approach
- **Model Improvements**:
  - Configured the timm model to use global average pooling
  - Added proper model weight initialization to match big_vision
  - Implemented sincos2d position embeddings to replace learned ones
  - Updated the loss function for MixUp

## File Structure

- `main.py`: Entry point for training
- `plain_vit.py`: Implementation of the ViT model with positional embedding modifications
- `utils/`
  - `__init__.py`: Imports for utility functions
  - `misc.py`: Various helper functions for distributed training, logging, etc.
  - `big_vision_randaugment.py`: Implementation of big_vision's RandAugment
  - `big_vision_twohotmixup.py`: Implementation of big_vision's MixUp approach
  - `lr_sched.py`: Learning rate scheduler implementation