import csv
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Nastaví seed pre reprodukovateľnosť výsledkov.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    """
    Vytvorí priečinok, ak neexistuje.
    """
    path.mkdir(parents=True, exist_ok=True)


def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Vypočíta accuracy pre batch.
    """
    preds = outputs.argmax(dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total


def compute_gradient_norm(model: torch.nn.Module) -> float:
    """
    Vypočíta L2 normu gradientov všetkých parametrov modelu.
    """
    total_norm = 0.0

    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2

    return total_norm ** 0.5


def get_memory_usage_mb(device: str) -> float:
    """
    Vráti aktuálne využitie GPU pamäte v MB.
    Ak sa používa CPU, vráti 0.
    """
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


def save_history_to_csv(history: list, filepath: Path) -> None:
    """
    Uloží históriu tréningu do CSV súboru.
    """
    if not history:
        return

    fieldnames = list(history[0].keys())

    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
