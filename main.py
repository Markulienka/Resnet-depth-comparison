import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from models import get_model
from train import evaluate, get_data_loaders, train_one_epoch
from utils import (
    ensure_dir,
    get_device,
    get_peak_memory_mb,
    reset_peak_memory,
    save_history_to_csv,
    set_seed,
)


def main() -> None:
    set_seed(config.SEED)

    ensure_dir(config.RESULTS_DIR)
    ensure_dir(config.BEST_MODELS_DIR)
    ensure_dir(config.LOGS_DIR)

    device = get_device()
    print(f"Použité zariadenie: {device}")

    train_loader, test_loader = get_data_loaders(device)

    model = get_model(config.MODEL_NAME, config.NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=config.ETA_MIN)

    history: list[dict[str, int | float]] = []
    best_val_acc = 0.0
    history_path = config.LOGS_DIR / f"{config.MODEL_NAME}_history.csv"

    total_start_time = time.time()

    for epoch in range(1, config.NUM_EPOCHS + 1):
        reset_peak_memory(device)
        epoch_start_time = time.time()

        train_loss, train_acc, grad_norm = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config.NUM_EPOCHS
        )
        val_loss, val_acc = evaluate(
            model, test_loader, criterion, device, epoch, config.NUM_EPOCHS
        )

        scheduler.step()

        epoch_time = time.time() - epoch_start_time
        peak_memory_mb = get_peak_memory_mb(device)

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_acc, 6),
            "gradient_norm": round(grad_norm, 6),
            "epoch_time_sec": round(epoch_time, 2),
            "peak_memory_mb": round(peak_memory_mb, 2),
        }
        history.append(row)
        save_history_to_csv(history, history_path)

        print(
            f"Epoch {epoch}/{config.NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
            f"grad_norm={grad_norm:.4f} | time={epoch_time:.2f}s | "
            f"peak_mem={peak_memory_mb:.2f}MB"
        )

        if config.SAVE_BEST_MODEL and val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = config.BEST_MODELS_DIR / f"{config.MODEL_NAME}_best.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_acc": val_acc,
            }, checkpoint_path)

    total_time = time.time() - total_start_time

    print("\nTréning dokončený.")
    print(f"Celkový čas: {total_time:.2f} s")
    print(f"História uložená do: {history_path}")


if __name__ == "__main__":
    main()