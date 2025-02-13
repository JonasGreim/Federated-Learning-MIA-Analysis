"""my-awesome-app: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
from flwr_datasets.visualization import plot_label_distributions
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


class Net(nn.Module):

    def __init__(self) -> None:
        super(Net, self).__init__()
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


def get_transforms():
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    return apply_transforms


fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        # Initialize FederatedDataset with Dirichlet partitioning
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={
                "train": DirichletPartitioner(
                    num_partitions=num_partitions,
                    partition_by="label",
                    alpha=0.3,  # Lower alpha -> more imbalanced partitions
                    seed=42,
                    min_partition_size=0,
                ),
            },
        )

        # Visualize label distribution across partitions
        partitioner = fds.partitioners["train"]
        fig, ax, df = plot_label_distributions(
            partitioner,
            label_name="label",
            plot_type="bar",
            size_unit="absolute",
            partition_id_axis="x",
            legend=True,
            verbose_labels=True,
            title="Per Partition Labels Distribution",
        )
        fig2, ax2, df2 = plot_label_distributions(
            partitioner,
            label_name="label",
            plot_type="heatmap",
            size_unit="absolute",
            partition_id_axis="x",
            legend=True,
            verbose_labels=True,
            title="Per Partition Labels Distribution",
            plot_kwargs={"annot": True},
        )
        fig.savefig("metrics_of_run/bar_label_distribution.png", dpi=300)
        fig2.savefig("metrics_of_run/heatmap_label_distribution.png", dpi=300)

    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    partition_train_test = partition_train_test.with_transform(get_transforms())
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


def train(net, trainloader, epochs, lr, device) -> tuple[float, float]:
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # (faster convergence but can overfit)
    net.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = net(images)  # Forward pass
            loss = criterion(outputs, labels)
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)  # Get predicted class
            correct += (predicted == labels).sum().item()  # Count correct predictions
            total += labels.size(0)  # Count total samples

    avg_trainloss = running_loss / len(trainloader)
    avg_trainacc = correct / total  # Average accuracy

    return avg_trainloss, avg_trainacc


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
