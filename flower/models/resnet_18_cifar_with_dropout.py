from torchvision import models
import torch.nn as nn
import torch


class ResNet18WithDropout(nn.Module):
    def __init__(self, p_head=0.5, p_block=0.15):
        super().__init__()
        m = models.resnet18(weights=None)
        m.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        m.maxpool = nn.Identity()
        self.layer1, self.layer2, self.layer3, self.layer4 = m.layer1, m.layer2, m.layer3, m.layer4
        self.drop2 = nn.Dropout2d(p_block)
        self.drop3 = nn.Dropout2d(p_block)
        self.avgpool = m.avgpool
        in_features = m.fc.in_features
        self.head_drop = nn.Dropout(p_head)
        self.fc = nn.Linear(in_features, 10)

    def forward(self, x):
        x = self.layer1(x)  # no extra ReLU
        x = self.drop2(self.layer2(x))  # Dropout2d on layer output
        x = self.drop3(self.layer3(x))
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head_drop(x)
        return self.fc(x)
