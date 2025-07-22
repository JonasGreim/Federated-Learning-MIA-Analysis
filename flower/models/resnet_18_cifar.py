import torch.nn as nn
from torchvision import models


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=False)

        # Modify the first conv layer for small CIFAR-10 pictures: 7x7 -> 3x3, stride 2 -> 1
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Remove maxpool to preserve spatial dimensions for small images
        self.model.maxpool = nn.Identity()

        # Replace the final fully connected layer to output 10 classes (CIFAR-10)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        return self.model(x)
