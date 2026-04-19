import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import config
from models import get_model
from utils import (
    calculate_accuracy,
    compute_gradient_norm,
    ensure_dir,
    get_memory_usage_mb,
    save_history_to_csv,
    set_seed,
)


def get_data_loaders():
    """
    Pripraví DataLoadery pre CIFAR-10.
    """
    train_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(config.CIFAR10_MEAN, config.CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.CIFAR10_MEAN, config.CIFAR10_STD),
    ])

    train_dataset = datasets.CIFAR10(
        root=str(config.PROJECT_ROOT / "data"),
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=str(config.PROJECT_ROOT / "data"),
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()

    running_loss = 0.0
    running_acc = 0.0
    running_grad_norm = 0.0
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

        acc = calculate_accuracy(outputs, labels)

        running_loss += loss.item()
        running_acc += acc
        running_grad_norm += grad_norm
        num_batches += 1

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{acc:.4f}",
            "grad": f"{grad_norm:.4f}",
        })

    return (
        running_loss / num_batches,
        running_acc / num_batches,
        running_grad_norm / num_batches,
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch, total_epochs):
    model.eval()

    running_loss = 0.0
    running_acc = 0.0
    num_batches = 0

    progress_bar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [eval]", leave=False)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        acc = calculate_accuracy(outputs, labels)

        running_loss += loss.item()
        running_acc += acc
        num_batches += 1

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{acc:.4f}",
        })

    return running_loss / num_batches, running_acc / num_batches


def main():
    set_seed(config.SEED)

    ensure_dir(config.RESULTS_DIR)
    ensure_dir(config.BEST_MODELS_DIR)

    device = config.DEVICE
    print(f"Použité zariadenie: {device}")

    train_loader, test_loader = get_data_loaders()

    model = get_model(config.MODEL_NAME, config.NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    history = []
    best_val_acc = 0.0

    total_start_time = time.time()

    for epoch in range(1, config.NUM_EPOCHS + 1):
        epoch_start_time = time.time()

        train_loss, train_acc, grad_norm = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config.NUM_EPOCHS
        )
        val_loss, val_acc = evaluate(
            model, test_loader, criterion, device, epoch, config.NUM_EPOCHS
        )

        epoch_time = time.time() - epoch_start_time
        memory_mb = get_memory_usage_mb(device)

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_acc, 6),
            "gradient_norm": round(grad_norm, 6),
            "epoch_time_sec": round(epoch_time, 2),
            "memory_mb": round(memory_mb, 2),
        }
        history.append(row)

        print(
            f"Epoch {epoch}/{config.NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
            f"grad_norm={grad_norm:.4f} | time={epoch_time:.2f}s | "
            f"memory={memory_mb:.2f}MB"
        )

        if config.SAVE_BEST_MODEL and val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = config.BEST_MODELS_DIR / f"{config.MODEL_NAME}_best.pth"
            torch.save(model.state_dict(), model_path)

    total_time = time.time() - total_start_time

    history_path = config.RESULTS_DIR / f"{config.MODEL_NAME}_history.csv"
    save_history_to_csv(history, history_path)

    print("\nTréning dokončený.")
    print(f"Celkový čas: {total_time:.2f} s")
    print(f"História uložená do: {history_path}")


if __name__ == "__main__":
    main()
