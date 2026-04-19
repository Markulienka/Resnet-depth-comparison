import torch.nn as nn
from torchvision import models


def get_model(model_name: str, num_classes: int = 10) -> nn.Module:
    """
    Vráti zvolený ResNet model upravený pre CIFAR-10.
    """
    model_name = model_name.lower()

    if model_name == "resnet34":
        model = models.resnet34(weights=None)
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"Neznámy model: {model_name}. Použi 'resnet34' alebo 'resnet50'.")

    # Nahradenie poslednej fully connected vrstvy
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
