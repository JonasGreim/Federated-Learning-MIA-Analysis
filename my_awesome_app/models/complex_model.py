import torch
import torch.nn as nn
import torch.nn.functional as F


class NetComplex(nn.Module):

    def __init__(self) -> None:
        super(NetComplex, self).__init__()
        # Convolutional Layers -> extract features from the image, kearnel_size: size of the filter, padding: add zeros to the border of the image
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # reducing internal covariate shift -> shift values to zero mean and unit variance (implicit form of regularization)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Pooling
        # reduce feature map, holds only the highest value of the 2x2 square -> 1 (half matrix), stride: how many steps the filter moves each time
        self.pool = nn.MaxPool2d(2, 2)

        # Fully Connected Layers
        # purpose: combine features + learn relationships between features -> flatten the data for prediction
        self.fc1 = nn.Linear(128 * 4 * 4, 256)  # Adjust based on image size
        self.bn4 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)

        # Dropout to prevent overfitting -> random subset of neurons is deactivated (in training)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """Compute forward pass."""
        # ReLu: fc1(x)=W⋅x+b -> apply weights to the input data, add bias, relu kinda sorts out unrelevant data (<0)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Additional Conv Layer

        x = torch.flatten(x, 1)  # Flatten for FC layers
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)  # Dropout for regularization
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)  # Another dropout layer
        x = self.fc3(x)  # Output layer -> 10 different classes
        return x
