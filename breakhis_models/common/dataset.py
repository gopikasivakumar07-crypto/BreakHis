import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size: int = 224):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def build_dataloaders(data_dir: str, img_size: int = 224, batch_size: int = 32,
                       num_workers: int = 8):
    train_tf, eval_tf = build_transforms(img_size)

    train_ds = datasets.ImageFolder(f"{data_dir}/train", transform=train_tf)
    val_ds = datasets.ImageFolder(f"{data_dir}/val", transform=eval_tf)
    test_ds = datasets.ImageFolder(f"{data_dir}/test", transform=eval_tf)

    assert train_ds.classes == val_ds.classes == test_ds.classes, \
        "Class folders differ between splits — check your directory structure."

    class_counts = torch.bincount(torch.tensor(train_ds.targets))
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[torch.tensor(train_ds.targets)]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    loader_kwargs = dict(num_workers=num_workers, pin_memory=True,
                          persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                               drop_last=True,
                               **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, train_ds.classes, class_weights
