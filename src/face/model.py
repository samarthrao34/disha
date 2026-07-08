"""Model factory for the DISHA face emotion pipeline."""

from typing import Literal

import torch.nn as nn
from torchvision import models

FaceModelName = Literal["resnet18", "mobilenet_v3_small", "mobilenet_v3_large"]


def create_resnet18_face_model(num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    """Create a ResNet-18 with the final layer replaced for emotion classes."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def create_mobilenet_v3_small_face_model(
    num_classes: int = 7,
    pretrained: bool = True,
) -> nn.Module:
    """Create MobileNetV3-Small with the classifier replaced for emotion classes."""
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def create_mobilenet_v3_large_face_model(
    num_classes: int = 7,
    pretrained: bool = True,
) -> nn.Module:
    """Create MobileNetV3-Large with the classifier replaced for emotion classes."""
    weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_large(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def create_face_model(
    model_name: FaceModelName = "resnet18",
    num_classes: int = 7,
    pretrained: bool = True,
) -> nn.Module:
    """Create a supported face-emotion model by name."""
    if model_name == "resnet18":
        return create_resnet18_face_model(num_classes=num_classes, pretrained=pretrained)

    if model_name == "mobilenet_v3_small":
        return create_mobilenet_v3_small_face_model(
            num_classes=num_classes,
            pretrained=pretrained,
        )

    if model_name == "mobilenet_v3_large":
        return create_mobilenet_v3_large_face_model(
            num_classes=num_classes,
            pretrained=pretrained,
        )

    raise ValueError(
        f"Unsupported face model: {model_name}. "
        "Choose from: resnet18, mobilenet_v3_small, mobilenet_v3_large."
    )