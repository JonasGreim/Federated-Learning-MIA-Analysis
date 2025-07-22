from flower.models.resnet_18_cifar import ResNet18
from flower.models.resnet_18_cifar_with_dropout import ResNet18WithDropout
from flower.models.shokri_simple_cnn import SimpleCnn
from flower.models.shokri_simple_cnn_with_dropout import SimpleCnnWithDropout
import torch.nn as nn


def create_model(model_name) -> nn.Module:
    if model_name == "simple_model":
        return SimpleCnn()
    elif model_name == "simple_model_with_dropout":
        return SimpleCnnWithDropout()
    elif model_name == "complex_model":
        return ResNet18()
    elif model_name == "complex_model_with_dropout":
        return ResNet18WithDropout()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
