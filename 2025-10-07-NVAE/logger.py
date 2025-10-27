import os
import sys
from datetime import datetime


class Logger:
    """日志记录器，同时输出到终端和文件"""
    
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

    def __enter__(self):
        self._prev_stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.stdout = getattr(self, '_prev_stdout', self.terminal)
        self.close()


def setup_experiment_dir(exp_name="nvae_cifar10"):
    """设置实验目录"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join("experiments", f"{exp_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # 创建子目录
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "output"), exist_ok=True)
    
    print(f"实验目录: {exp_dir}")
    return exp_dir