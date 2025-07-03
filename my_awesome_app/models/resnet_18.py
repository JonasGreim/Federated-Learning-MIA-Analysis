from torchvision import models
import torch.nn as nn


def create_resnet18_model() -> nn.Module:
    # Load pretrained ResNet-18 (optional: pretrained=False if you don't want ImageNet weights)
    model = models.resnet18(pretrained=False)

    # Adjust the final fully connected layer to output 10 classes for CIFAR-10
    model.fc = nn.Linear(model.fc.in_features, 10)

    return model
