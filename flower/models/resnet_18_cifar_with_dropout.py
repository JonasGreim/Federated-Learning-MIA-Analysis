from torchvision import models
import torch.nn as nn
import torch


class ResNet18WithDropout(nn.Module):
    def __init__(self, p_head=0.5, p_block=0.15):
        super(ResNet18WithDropout, self).__init__()
        self.model = models.resnet18(weights=None)

        # CIFAR modifications
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

        # --- Dropouts (only additions) ---
        self.drop2 = nn.Dropout2d(p_block)
        self.drop3 = nn.Dropout2d(p_block)
        self.head_drop = nn.Dropout(p_head)

    def forward(self, x):
        # identical to torchvision ResNet18 forward, with dropout inserted
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.drop2(self.model.layer2(x))
        x = self.drop3(self.model.layer3(x))
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head_drop(x)
        x = self.model.fc(x)
        return x
