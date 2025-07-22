import torch
import torch.nn as nn


# CNN defined in the paper "Membership Inference Attacks Against Machine Learning Models", by Shokri et al. (2017)
# Note: Paper uses Torch7; here we are using PyTorch
# conv_filters (32,64 ), kernel_size and padding & pooling kernel size and stride are not descripted (only descripted as standard CNN)
class SimpleCnn(nn.Module):
    def __init__(self):
        super(SimpleCnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.tanh(self.fc1(x))
        return self.fc2(x)  # no softmax in the last layer, as it is not needed for the loss function (CrossEntropyLoss)
