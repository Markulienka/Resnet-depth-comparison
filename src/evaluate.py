import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from models import get_model
from utils import ensure_dir

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def load_checkpoint(run_name: str, model_name: str, device: str) -> torch.nn.Module:
    checkpoint_path = config.BEST_MODELS_DIR / f"{run_name}_best.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint nenájdený: {checkpoint_path}")

    model = get_model(model_name, config.NUM_CLASSES).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[tuple[int, int]]]:
    all_preds = []
    all_labels = []
    all_images = []
    all_image_labels: list[tuple[int, int]] = []

    for images, labels in loader:
        images = images.to(device)
        images_cpu = images.cpu()
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        labels_np = labels.numpy()

        all_preds.append(preds)
        all_labels.append(labels_np)

        wrong_mask = preds != labels_np
        for i, wrong in enumerate(wrong_mask):
            if wrong:
                all_images.append(images_cpu[i].numpy())
                all_image_labels.append((int(labels_np[i]), int(preds[i])))

    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        all_images,
        all_image_labels,
    )


def compute_confusion_matrix(
    preds: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(matrix, (labels, preds), 1)
    return matrix


def save_confusion_matrix_plot(
    matrix: np.ndarray,
    run_name: str,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(config.NUM_CLASSES))
    ax.set_yticks(range(config.NUM_CLASSES))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CIFAR10_CLASSES)
    ax.set_xlabel("Predikovaná trieda")
    ax.set_ylabel("Skutočná trieda")
    ax.set_title(f"Confusion Matrix — {run_name}")

    thresh = matrix.max() / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, str(matrix[i, j]),
                ha="center", va="center",
                color="white" if matrix[i, j] > thresh else "black",
                fontsize=8,
            )

    plt.tight_layout()
    out_path = output_dir / f"{run_name}_confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Confusion matrix uložená: {out_path}")


def save_per_class_accuracy(
    matrix: np.ndarray,
    run_name: str,
    output_dir: Path,
) -> None:
    rows = []
    for i, class_name in enumerate(CIFAR10_CLASSES):
        total = matrix[i].sum()
        correct = matrix[i][i]
        acc = correct / total if total > 0 else 0.0
        rows.append({"class": class_name, "correct": correct, "total": total, "accuracy": round(acc, 4)})

    rows.sort(key=lambda r: r["accuracy"])

    out_path = output_dir / f"{run_name}_per_class_accuracy.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "correct", "total", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nPer-class accuracy ({run_name}):")
    for r in rows:
        print(f"  {r['class']:<12} {r['accuracy']:.2%}  ({r['correct']}/{r['total']})")

    print(f"CSV uložené: {out_path}")


def denormalize(image: np.ndarray) -> np.ndarray:
    mean = np.array(config.CIFAR10_MEAN).reshape(3, 1, 1)
    std = np.array(config.CIFAR10_STD).reshape(3, 1, 1)
    image = image * std + mean
    return np.clip(image.transpose(1, 2, 0), 0, 1)


def save_misclassified_samples(
    images: list[np.ndarray],
    labels: list[tuple[int, int]],
    run_name: str,
    output_dir: Path,
    max_samples: int = 20,
) -> None:
    if not images:
        print("Žiadne nesprávne klasifikácie nenájdené.")
        return

    samples = list(zip(images, labels))[:max_samples]
    cols = 5
    rows = (len(samples) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = np.array(axes).reshape(-1)

    for i, (img, (true_label, pred_label)) in enumerate(samples):
        axes[i].imshow(denormalize(img))
        axes[i].set_title(
            f"True: {CIFAR10_CLASSES[true_label]}\nPred: {CIFAR10_CLASSES[pred_label]}",
            fontsize=8,
            color="red",
        )
        axes[i].axis("off")

    for j in range(len(samples), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Nesprávne klasifikácie — {run_name}", fontsize=12)
    plt.tight_layout()

    out_path = output_dir / f"{run_name}_misclassified.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Nesprávne klasifikácie uložené: {out_path}")


def evaluate_model(
    run_name: str,
    model_name: str,
    test_loader: torch.utils.data.DataLoader,
    device: str,
) -> float:
    output_dir = config.RESULTS_DIR / "analysis"
    ensure_dir(output_dir)

    print(f"\n=== Analýza: {run_name} ===")

    model = load_checkpoint(run_name, model_name, device)
    preds, labels, wrong_images, wrong_labels = collect_predictions(model, test_loader, device)

    total_acc = (preds == labels).mean()
    print(f"Test accuracy: {total_acc:.4%}")
    print(f"Nesprávne klasifikácií: {(preds != labels).sum()} / {len(labels)}")

    matrix = compute_confusion_matrix(preds, labels, config.NUM_CLASSES)
    save_confusion_matrix_plot(matrix, run_name, output_dir)
    save_per_class_accuracy(matrix, run_name, output_dir)
    save_misclassified_samples(wrong_images, wrong_labels, run_name, output_dir)
    return float(total_acc)