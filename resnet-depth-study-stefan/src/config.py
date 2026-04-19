from pathlib import Path
import torch

# Základné nastavenia projektu
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
BEST_MODELS_DIR = RESULTS_DIR / "best_models"

# Model: "resnet34" alebo "resnet50"
MODEL_NAME = "resnet34"

# Dataset / tréning
NUM_CLASSES = 10
BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
IMAGE_SIZE = 32  # ResNet modely bežne fungujú komfortnejšie s väčším vstupom

# Reprodukovateľnosť
SEED = 42

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset štatistiky pre CIFAR-10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# Ukladanie
SAVE_BEST_MODEL = True
