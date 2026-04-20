import csv
import sys
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt

import config
from utils import ensure_dir


class MetricConfig(NamedTuple):
    train_key: str
    val_key: str | None
    title: str
    ylabel: str


METRICS: list[MetricConfig] = [
    MetricConfig("train_loss", "val_loss", "Loss", "Loss"),
    MetricConfig("train_accuracy", "val_accuracy", "Accuracy", "Accuracy"),
    MetricConfig("gradient_norm", None, "Gradient Norm", "L2 Norm"),
]


def load_history(run_name: str) -> list[dict[str, str]]:
    path = config.LOGS_DIR / f"{run_name}_history.csv"
    if not path.exists():
        raise FileNotFoundError(f"História nenájdená: {path}")

    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_metric_comparison(
    histories: dict[str, list[dict[str, str]]],
    train_key: str,
    val_key: str | None,
    title: str,
    ylabel: str,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    for run_name, history in histories.items():
        try:
            epochs = [int(row["epoch"]) for row in history]
        except KeyError:
            raise KeyError(f"Stĺpec 'epoch' nenájdený v histórii '{run_name}'") from None

        try:
            train_vals = [float(row[train_key]) for row in history]
        except KeyError:
            raise KeyError(f"Stĺpec '{train_key}' nenájdený v histórii '{run_name}'") from None

        label = f"{run_name} train" if val_key else run_name
        ax.plot(epochs, train_vals, label=label, linewidth=1.5)

        if val_key:
            try:
                val_vals = [float(row[val_key]) for row in history]
            except KeyError:
                raise KeyError(f"Stĺpec '{val_key}' nenájdený v histórii '{run_name}'") from None
            ax.plot(epochs, val_vals, label=f"{run_name} val", linewidth=1.5, linestyle="--")

    ax.set_title(title)
    ax.set_xlabel("Epocha")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / f"comparison_{train_key}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Graf uložený: {out_path}")


def plot_all(run_names: list[str]) -> None:
    output_dir = config.RESULTS_DIR / "plots"
    ensure_dir(output_dir)

    histories: dict[str, list[dict[str, str]]] = {}
    for run_name in run_names:
        try:
            histories[run_name] = load_history(run_name)
        except FileNotFoundError as e:
            print(f"Preskočené: {e}")

    if not histories:
        print("Žiadne histórie na vykreslenie.")
        return

    for metric in METRICS:
        plot_metric_comparison(
            histories, metric.train_key, metric.val_key,
            metric.title, metric.ylabel, output_dir,
        )