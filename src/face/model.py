"""Model factory for the DISHA face emotion pipeline."""

import torch.nn as nn
from torchvision import models


def create_resnet18_face_model(num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    """Create a ResNet-18 with the final layer replaced for emotion classes.

    Args:
        num_classes: Number of emotion classes (7 for FER2013).
        pretrained: If True, initialize with ImageNet weights.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
