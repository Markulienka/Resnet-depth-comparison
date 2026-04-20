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
    norms = [p.grad.detach().norm(2) ** 2 for p in model.parameters() if p.grad is not None]
    return torch.stack(norms).sum().sqrt().item() if norms else 0.0


def get_peak_memory_mb(device: str) -> float:
    """
    Vráti peak využitie GPU pamäte v MB od posledného resetu.
    Pre MPS vracia aktuálne alokovanú pamäť (peak API nie je dostupné).
    Pre CPU vráti 0.
    """
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    if device == "mps":
        return torch.mps.current_allocated_memory() / (1024 ** 2)
    return 0.0


def reset_peak_memory(device: str) -> None:
    """
    Resetuje štatistiku peak GPU pamäte pred novou epochou.
    Pre MPS reset nie je dostupný — štatistika sa neaktualizuje.
    """
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


# --- Výstup ---

def append_history_row(
    row: dict[str, int | float],
    filepath: Path | str,
) -> None:
    """
    Pripíše jeden riadok do CSV histórie.
    Hlavičku zapíše automaticky ak súbor neexistuje alebo je prázdny.
    """
    filepath = Path(filepath)
    write_header = not filepath.exists() or filepath.stat().st_size == 0
    mode = "w" if write_header else "a"
    with open(filepath, mode=mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)