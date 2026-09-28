"""Dataset loaders for SSL pretraining and downstream evaluation."""

import os
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as tv_datasets

from .augmentations import SSLAugmentation, get_eval_transform


class SSLDataset(Dataset):
    """Wraps a torchvision dataset with SSL augmentation (two views)."""

    def __init__(self, base_dataset, transform: SSLAugmentation):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        view1, view2 = self.transform(img)
        return view1, view2, label


def get_ssl_dataloader(
    dataset: str = "cifar100",
    data_dir: str = "./data",
    batch_size: int = 256,
    num_workers: int = 2,
) -> DataLoader:
    """Get SSL pretraining dataloader (returns two views + label)."""
    transform = SSLAugmentation(dataset=dataset)

    if dataset == "cifar100":
        base = tv_datasets.CIFAR100(data_dir, train=True, download=True)
    elif dataset == "stl10":
        # Use train+unlabeled split for SSL
        base = tv_datasets.STL10(data_dir, split="train+unlabeled", download=True)
    elif dataset == "imagenet100":
        # ImageNet-100: expects ImageFolder structure at data_dir/imagenet100/train/
        train_dir = os.path.join(data_dir, "imagenet100", "train")
        if not os.path.exists(train_dir):
            raise FileNotFoundError(
                f"ImageNet-100 not found at {train_dir}. "
                "Please download and organize into ImageFolder format."
            )
        base = tv_datasets.ImageFolder(train_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    ds = SSLDataset(base, transform)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )


def get_eval_dataloaders(
    dataset: str = "cifar100",
    data_dir: str = "./data",
    batch_size: int = 256,
    num_workers: int = 2,
) -> tuple:
    """Get train/test dataloaders for linear evaluation (standard transforms)."""
    transform = get_eval_transform(dataset)

    if dataset == "cifar100":
        train_ds = tv_datasets.CIFAR100(data_dir, train=True, download=True, transform=transform)
        test_ds = tv_datasets.CIFAR100(data_dir, train=False, download=True, transform=transform)
    elif dataset == "stl10":
        train_ds = tv_datasets.STL10(data_dir, split="train", download=True, transform=transform)
        test_ds = tv_datasets.STL10(data_dir, split="test", download=True, transform=transform)
    elif dataset == "imagenet100":
        train_dir = os.path.join(data_dir, "imagenet100", "train")
        val_dir = os.path.join(data_dir, "imagenet100", "val")
        train_ds = tv_datasets.ImageFolder(train_dir, transform=transform)
        test_ds = tv_datasets.ImageFolder(val_dir, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader
