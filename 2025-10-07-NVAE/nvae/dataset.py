import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    
    def __init__(self, root='./data', img_dim=32, **kwargs):
        self.img_dim = (img_dim, img_dim) if isinstance(img_dim, int) else img_dim
        self.dataset = self._create_dataset(root, **kwargs)
    
    def _create_dataset(self, root, **kwargs):
        """子类需要实现此方法"""
        raise NotImplementedError
    
    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image
    
    def __len__(self):
        return len(self.dataset)


class CIFAR10Dataset(BaseDataset):
    
    def _create_dataset(self, root, train=True, download=True, **kwargs):
        transform = transforms.Compose([transforms.ToTensor()])
        return torchvision.datasets.CIFAR10(
            root=root, train=train, download=download, transform=transform
        )


class CelebADataset(BaseDataset):
    
    def _create_dataset(self, root, split='train', download=False, **kwargs):
        transform = transforms.Compose([
            transforms.Resize(self.img_dim),
            transforms.CenterCrop(self.img_dim),
            transforms.ToTensor(),
        ])
        return torchvision.datasets.CelebA(
            root=root, split=split, download=download, transform=transform
        )
