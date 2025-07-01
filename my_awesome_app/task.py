"""my-awesome-app: A Flower / PyTorch app."""

from collections import OrderedDict
import os
import torch
from flwr_datasets.visualization import plot_label_distributions
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torchvision.transforms import Compose, Normalize, ToTensor
import toml
from my_awesome_app.models.complex_model import NetComplex
from my_awesome_app.models.simple_model import NetSimple
from flwr.common import Metrics
from typing import List, Tuple
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset, DataLoader
from torchvision import transforms
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")


def create_model() -> nn.Module:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pyproject_path = os.path.join(current_dir, '..', 'pyproject.toml')

    with open(pyproject_path) as file:
        data = toml.load(file)
    # Access the model name from the configuration file
    model_name: str = data["tool"]["flwr"]["app"]["config"]["model"]

    if model_name == "complex_model":
        return NetComplex()
    elif model_name == "simple_model":
        return NetSimple()
    else:
        raise ValueError(f"Unknown model name: {model_name}")


# def get_transforms():
#     pytorch_transforms = Compose(
#         [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#     )
#
#     def apply_transforms(batch):
#         """Apply transforms to the partition from FederatedDataset."""
#         batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
#         return batch
#
#     return apply_transforms


def get_transforms_custom():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_flower_partition(data_indices):
    # Return only a subset of CIFAR10
    os.makedirs(DATA_DIR, exist_ok=True)
    full_dataset = CIFAR10(root=DATA_DIR, train=True, download=True, transform=get_transforms_custom())
    return Subset(full_dataset, data_indices)


def load_data_custom(partition_id: int, num_partitions: int, indices: list):
    # Handle edge case if data doesn't divide evenly
    partition_sizes = np.array_split(indices, num_partitions)
    client_indices = partition_sizes[partition_id]

    train_size = int(0.8 * len(client_indices))
    train_idx = client_indices[:train_size]
    test_idx = client_indices[train_size:]

    trainset = get_flower_partition(train_idx)
    testset = get_flower_partition(test_idx)

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32)
    return trainloader, testloader


fds = None  # Cache FederatedDataset


# def load_data(partition_id: int, num_partitions: int):
#     """Load partition CIFAR10 data."""
#     # Only initialize `FederatedDataset` once
#     global fds
#     if fds is None:
#         # Initialize FederatedDataset with Dirichlet partitioning
#         fds = FederatedDataset(
#             dataset="uoft-cs/cifar10",
#             partitioners={
#                 "train": DirichletPartitioner(
#                     num_partitions=num_partitions,
#                     partition_by="label",
#                     alpha=0.3,  # Lower alpha -> more imbalanced partitions
#                     seed=42,
#                     min_partition_size=0,
#                 ),
#             },
#         )
#
#         # Visualize label distribution across partitions
#         # partitioner = fds.partitioners["train"]
#         # fig, ax, df = plot_label_distributions(
#         #     partitioner,
#         #     label_name="label",
#         #     plot_type="bar",
#         #     size_unit="absolute",
#         #     partition_id_axis="x",
#         #     legend=True,
#         #     verbose_labels=True,
#         #     title="Per Partition Labels Distribution",
#         # )
#         # fig2, ax2, df2 = plot_label_distributions(
#         #     partitioner,
#         #     label_name="label",
#         #     plot_type="heatmap",
#         #     size_unit="absolute",
#         #     partition_id_axis="x",
#         #     legend=True,
#         #     verbose_labels=True,
#         #     title="Per Partition Labels Distribution",
#         #     plot_kwargs={"annot": True},
#         # )
#         # fig.savefig("metrics_of_run/bar_label_distribution.png", dpi=300)
#         # fig2.savefig("metrics_of_run/heatmap_label_distribution.png", dpi=300)
#
#     partition = fds.load_partition(partition_id)
#     # Divide data on each node: 80% train, 20% test
#     partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
#
#     partition_train_test = partition_train_test.with_transform(get_transforms())
#     trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
#     testloader = DataLoader(partition_train_test["test"], batch_size=32)
#     return trainloader, testloader


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
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)

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
    avg_trainacc = correct / total  # Average accuracy over epochs

    return avg_trainloss, avg_trainacc


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
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


def get_dataset_split_flag():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pyproject_path = os.path.join(current_dir, "..", "pyproject.toml")
    with open(pyproject_path) as f:
        config = toml.load(f)
    return config["tool"]["flwr"]["app"]["config"].get("dataset-split")
