"""my-awesome-app: A Flower / PyTorch app."""

from collections import OrderedDict
import os
from typing import List
import random
import torch
import torch.nn as nn
import toml
from datasets import load_from_disk
from flwr_datasets.partitioner import DirichletPartitioner
from flwr_datasets.visualization import plot_label_distributions
from my_awesome_app.models.complex_model import NetComplex
from my_awesome_app.models.mia_paper_target_shadow_model import SimpleCNN
from my_awesome_app.models.resnet_18 import create_resnet18_model
from my_awesome_app.models.simple_model import NetSimple
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset, DataLoader
from torchvision import transforms
import numpy as np
from sklearn.utils import check_random_state
from my_awesome_app.utils.huggingface_to_pytorch import HFDatasetToTorch
from my_awesome_app.utils.wandb_logging_target import visualize_label_distribution

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
current_dir = os.path.dirname(os.path.abspath(__file__))
pyproject_path = os.path.join(current_dir, '..', '..', 'pyproject.toml')


def create_model(model_name) -> nn.Module:
    if model_name == "complex_model":
        return NetComplex()
    elif model_name == "simple_model":
        return NetSimple()
    elif model_name == "mia_paper":
        return SimpleCNN()
    elif model_name == "resnet18":
        return create_resnet18_model()
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def get_transforms_custom() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # Normalize with CIFAR-10 mean and std (shokri does not use normalization -> better overfitting)
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])


def get_flower_partition(data_indices) -> Subset:
    # Return only a subset of CIFAR10
    os.makedirs(DATA_DIR, exist_ok=True)
    full_dataset = CIFAR10(root=DATA_DIR, train=True, download=True, transform=get_transforms_custom())
    return Subset(full_dataset, data_indices)


def load_data_custom(partition_id: int, num_partitions: int, split: str, batch_size: int, seed: int) -> tuple[
    DataLoader, DataLoader]:

    try:
        split_dataset = load_from_disk(split)
    except FileNotFoundError:
        print(f"Error: The dataset at path '{split}' was not found.")
        split_dataset = None

    # Apply Dirichlet partitioning to the client data subset
    partitioner = DirichletPartitioner(
        num_partitions=num_partitions,
        partition_by="label",
        alpha=1.0,
        min_partition_size=0,
    )
    partitioner.dataset = split_dataset
    client_dataset = partitioner.load_partition(partition_id=partition_id)

    visualize_label_distribution(partitioner, "metrics_of_run")

    data_split = client_dataset.train_test_split(test_size=0.2, seed=seed)

    transform = get_transforms_custom()
    trainset = HFDatasetToTorch(data_split["train"], transform=transform)
    testset = HFDatasetToTorch(data_split["test"], transform=transform)

    g = torch.Generator()
    g.manual_seed(seed)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g,
                             num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"[Client {partition_id}] Loaded {len(trainset)} train samples, {len(testset)} test samples.")
    return trainloader, testloader


# fds = None  # Cache FederatedDataset
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
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    net.train()

    with open(pyproject_path) as file:
        data = toml.load(file)
    learning_rate_decay = data["tool"]["flwr"]["app"]["config"]["learning-rate-decay"]

    # Learning rate decay matching paper: decay = 1e-7
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda e: 1 / (1 + learning_rate_decay * e)
    )

    running_loss = 0.0
    correct = 0
    total = 0

    for epoch in range(epochs):
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

        # Apply learning rate decay at end of each epoch
        scheduler.step()

    avg_trainloss = running_loss / (epochs * len(trainloader))
    avg_trainacc = correct / total  # Average accuracy over epochs

    release_model(net, device.type)
    return avg_trainloss, avg_trainacc


def test(net, testloader, device) -> tuple[float, float]:
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
    accuracy = correct / len(testloader.dataset) if len(testloader.dataset) > 0 else 0.0
    loss = loss / len(testloader)
    release_model(net, device.type)
    return loss, accuracy


def get_weights(net: nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net: nn.Module, parameters: List[np.ndarray]) -> None:
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_dataset_split_flag() -> str:
    with open(pyproject_path) as f:
        config = toml.load(f)
    return config["tool"]["flwr"]["app"]["config"].get("dataset-split", "target")


def seed_everything(seed: int = 42) -> None:
    """Seed all major libraries and environment settings for full reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)  # Python hash
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # CUDA deterministic
    random.seed(seed)  # Python built-in RNG
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # Torch CPU
    torch.cuda.manual_seed(seed)  # Torch current GPU
    torch.cuda.manual_seed_all(seed)  # All GPUs
    torch.backends.cudnn.deterministic = True  # Force determinism
    torch.backends.cudnn.benchmark = False  # Disable auto-tuning
    torch.use_deterministic_algorithms(True)

    try:
        _ = check_random_state(seed)
    except ImportError:
        pass

    print(f"[Seed] Set global seed to {seed}")


def seed_worker(worker_id) -> None:
    # worker_id is automatically used internally by PyTorch to generate a unique seed for each worker process
    worker_seed = torch.initial_seed() % 2 ** 32  # worker-specific seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def release_model(model, device: str) -> None:
    """
    Safely release a PyTorch model from GPU memory.

    Moves the model to CPU, deletes the object, and clears GPU cache if needed.
    """
    if model is not None:
        model.cpu()
        del model
    if device == 'cuda':
        torch.cuda.empty_cache()
