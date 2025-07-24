from torchvision import models
import torch.nn as nn


class ResNet18WithDropout(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.model = models.resnet18(weights=None)

        # Adjust the first convolution for CIFAR-10
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Remove maxpool layer
        self.model.maxpool = nn.Identity()

        # Save input features to the final fc
        in_features = self.model.fc.in_features

        # Remove original fc, we will add dropout before it
        self.model.fc = nn.Identity()

        # Add dropout and final classification layer
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(in_features, 10)

    def forward(self, x):
        x = self.model(x)  # forward through ResNet blocks
        x = self.dropout(x)  # apply dropout
        x = self.fc(x)  # final classification
        return x
