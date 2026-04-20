import csv
import sys
import time
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import config
from evaluate import evaluate_model
from models import get_model
from plot import plot_all
from train import evaluate as evaluate_epoch, get_data_loaders, train_one_epoch
from utils import (
    append_history_row,
    ensure_dir,
    get_device,
    get_peak_memory_mb,
    reset_peak_memory,
    set_seed,
)


class Experiment(TypedDict):
    model_name: str
    lr: float
    run_name: str


class SummaryRow(TypedDict):
    run_name: str
    model: str
    lr: float
    best_val_acc: float
    test_acc: float


EXPERIMENTS: list[Experiment] = [
    {"model_name": "resnet34", "lr": 1e-3, "run_name": "resnet34_lr1e3"},
    {"model_name": "resnet34", "lr": 1e-4, "run_name": "resnet34_lr1e4"},
    {"model_name": "resnet50", "lr": 1e-3, "run_name": "resnet50_lr1e3"},
    {"model_name": "resnet50", "lr": 1e-4, "run_name": "resnet50_lr1e4"},
]


def run_experiment(
    model_name: str,
    run_name: str,
    lr: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
) -> float:
    print(f"\n{'=' * 60}")
    print(f"Experiment: {run_name}  |  LR={lr}")
    print(f"{'=' * 60}")

    model = get_model(model_name, config.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=config.ETA_MIN)

    best_val_acc = -1.0
    history_path = config.LOGS_DIR / f"{run_name}_history.csv"

    total_start = time.time()

    for epoch in range(1, config.NUM_EPOCHS + 1):
        reset_peak_memory(device)
        epoch_start = time.time()

        train_loss, train_acc, grad_norm = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config.NUM_EPOCHS
        )
        val_loss, val_acc = evaluate_epoch(
            model, val_loader, criterion, device, epoch, config.NUM_EPOCHS
        )

        scheduler.step()

        epoch_time = time.time() - epoch_start
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
        append_history_row(row, history_path)

        print(
            f"Epoch {epoch}/{config.NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
            f"grad_norm={grad_norm:.4f} | time={epoch_time:.2f}s | "
            f"peak_mem={peak_memory_mb:.2f}MB"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = config.BEST_MODELS_DIR / f"{run_name}_best.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_acc": val_acc,
            }, checkpoint_path)

    total_time = time.time() - total_start
    print(f"\n{run_name} hotový | čas={total_time:.2f}s | best_val_acc={best_val_acc:.4f}")
    return best_val_acc


def main() -> None:
    set_seed(config.SEED)

    ensure_dir(config.RESULTS_DIR)
    ensure_dir(config.BEST_MODELS_DIR)
    ensure_dir(config.LOGS_DIR)

    device = get_device()
    print(f"Použité zariadenie: {device}")

    train_loader, val_loader, test_loader = get_data_loaders(device)
    print(
        f"Dáta: train={len(train_loader.dataset)}, "
        f"val={len(val_loader.dataset)}, "
        f"test={len(test_loader.dataset)}"
    )

    summary: list[SummaryRow] = []

    print("\n\n=== TRÉNING A ANALÝZA CHÝB ===")
    for exp in EXPERIMENTS:
        best_val_acc = run_experiment(
            model_name=exp["model_name"],
            run_name=exp["run_name"],
            lr=exp["lr"],
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )
        test_acc = evaluate_model(
            run_name=exp["run_name"],
            model_name=exp["model_name"],
            test_loader=test_loader,
            device=device,
        )
        summary.append(
            {
                "run_name": exp["run_name"],
                "model": exp["model_name"],
                "lr": exp["lr"],
                "best_val_acc": round(best_val_acc, 6),
                "test_acc": round(test_acc, 6),
            }
        )

    print("\n\n=== GRAFY ===")
    plot_all([exp["run_name"] for exp in EXPERIMENTS])

    summary_path = config.RESULTS_DIR / "results_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(SummaryRow.__annotations__.keys()),
        )
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nSúhrn výsledkov: {summary_path}")

    print("\n--- Výsledky experimentov ---")
    for row in sorted(summary, key=lambda r: r["test_acc"], reverse=True):
        print(
            f"  {row['run_name']:<25} "
            f"val_acc={row['best_val_acc']:.4f} "
            f"test_acc={row['test_acc']:.4f}"
        )

    print("\nVšetko hotové.")


if __name__ == "__main__":
    main()