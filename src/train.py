import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

import config
from utils import compute_gradient_norm


def get_data_loaders(device: str) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Pripraví DataLoadery pre CIFAR-10.
    Vracia (train_loader, val_loader, test_loader).
    Val split je oddelený od test setu — test sa používa len na finálnu evaluáciu.
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(config.IMAGE_SIZE, padding=config.CROP_PADDING),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(config.CIFAR10_MEAN, config.CIFAR10_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.CIFAR10_MEAN, config.CIFAR10_STD),
    ])

    train_base = datasets.CIFAR10(
        root=str(config.DATA_DIR),
        train=True,
        download=True,
        transform=train_transform,
    )
    val_base = datasets.CIFAR10(
        root=str(config.DATA_DIR),
        train=True,
        download=False,
        transform=eval_transform,
    )

    val_size = int(len(train_base) * config.VAL_SPLIT)
    all_indices = torch.randperm(
        len(train_base),
        generator=torch.Generator().manual_seed(config.SEED),
    ).tolist()
    val_indices = all_indices[:val_size]
    train_indices = all_indices[val_size:]

    train_dataset = Subset(train_base, train_indices)
    val_dataset = Subset(val_base, val_indices)

    test_dataset = datasets.CIFAR10(
        root=str(config.DATA_DIR),
        train=False,
        download=True,
        transform=eval_transform,
    )

    pin = device.startswith("cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin,
    )

    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: str,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_grad_norm = 0.0
    num_batches = 0

    progress_bar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [train]", leave=True)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        grad_norm = compute_gradient_norm(model)
        optimizer.step()

        preds = outputs.argmax(dim=1)
        batch_correct = (preds == labels).sum().item()
        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += batch_correct
        total_samples += batch_size
        total_grad_norm += grad_norm
        num_batches += 1

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{batch_correct / batch_size:.4f}",
            "grad": f"{grad_norm:.4f}",
        })

    return (
        total_loss / total_samples,
        total_correct / total_samples,
        total_grad_norm / num_batches,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [eval]", leave=False)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        batch_correct = (preds == labels).sum().item()
        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += batch_correct
        total_samples += batch_size

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{batch_correct / batch_size:.4f}",
        })

    return total_loss / total_samples, total_correct / total_samples