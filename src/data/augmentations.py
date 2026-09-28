"""SSL augmentation pipelines.

Follows SimCLR augmentation protocol: random crop + flip + color jitter +
grayscale + gaussian blur. Returns two augmented views of the same image.
"""

import torchvision.transforms as T


# Dataset-specific image sizes
_IMG_SIZES = {
    "cifar100": 32,
    "stl10": 96,
    "imagenet100": 224,
}


class SSLAugmentation:
    """Returns two augmented views of the same image.

    Usage:
        transform = SSLAugmentation(dataset="cifar100")
        view1, view2 = transform(image)
    """

    def __init__(self, dataset: str = "cifar100"):
        img_size = _IMG_SIZES.get(dataset, 32)
        self.transform = _build_ssl_transform(img_size, dataset)

    def __call__(self, x):
        return self.transform(x), self.transform(x)


def _build_ssl_transform(img_size: int, dataset: str) -> T.Compose:
    """Build SimCLR-style augmentation pipeline."""

    # Smaller images (CIFAR) need gentler augmentation
    if dataset == "cifar100":
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.5071, 0.4867, 0.4408],
                        std=[0.2675, 0.2565, 0.2761]),
        ])
    elif dataset == "stl10":
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.4408, 0.4279, 0.3867],
                        std=[0.2682, 0.2610, 0.2686]),
        ])
    else:  # imagenet100
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])


def get_eval_transform(dataset: str = "cifar100") -> T.Compose:
    """Deterministic transform for evaluation (no augmentation)."""
    img_size = _IMG_SIZES.get(dataset, 32)
    normalize = {
        "cifar100": T.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
        "stl10": T.Normalize([0.4408, 0.4279, 0.3867], [0.2682, 0.2610, 0.2686]),
        "imagenet100": T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    }

    transforms = [T.Resize(img_size + 4), T.CenterCrop(img_size), T.ToTensor()]
    if dataset in normalize:
        transforms.append(normalize[dataset])
    return T.Compose(transforms)
