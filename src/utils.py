import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# --- Device ---

def get_device() -> str:
    """
    Zistí dostupné zariadenie: CUDA > MPS > CPU.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --- Reprodukovateľnosť ---

def set_seed(seed: int = 42) -> None:
    """
    Nastaví seed pre reprodukovateľnosť výsledkov.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- Súborový systém ---

def ensure_dir(path: Path | str) -> None:
    """
    Vytvorí priečinok, ak neexistuje.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


# --- Tréning ---

def compute_gradient_norm(model: nn.Module) -> float:
    """
    Vypočíta L2 normu gradientov všetkých parametrov modelu.
    """
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.norm(2).item()
            total_norm += param_norm ** 2
    return total_norm ** 0.5


def get_peak_memory_mb(device: str) -> float:
    """
    Vráti peak využitie GPU pamäte v MB od posledného resetu.
    Ak sa používa CPU, vráti 0.
    """
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    if device == "mps":
        return torch.mps.current_allocated_memory() / (1024 ** 2)
    return 0.0


def reset_peak_memory(device: str) -> None:
    """
    Resetuje štatistiku peak GPU pamäte pred novou epochou.
    """
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


# --- Výstup ---

def save_history_to_csv(history: list[dict[str, int | float]], filepath: Path | str) -> None:
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