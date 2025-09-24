from datasets import load_from_disk
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from torch.utils.data import DataLoader
from torchvision import transforms
from flower.utils.huggingface_to_pytorch import HFDatasetToTorch
from pathlib import Path
from flower.utils.reproducibility import seed_worker
import torch
from flower.utils.split_cifar10_mia import split_cifar10_for_target_and_shadow
from path_settings import D1_SPLIT_PATH, D2_SPLIT_PATH, D3_SPLIT_PATH, D4_SPLIT_PATH
import multiprocessing as mp


def get_transforms_custom() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # Normalize with CIFAR-10 mean and std
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])


def load_data_custom(iid: bool, dirichlet_alpha: float, partition_id: int, num_partitions: int, split: Path,
                     batch_size: int, seed: int) -> tuple[DataLoader, DataLoader]:
    ensure_split_data_exists()

    try:
        split_dataset = load_from_disk(split)
    except Exception as e:
        raise RuntimeError(f"Target model: Failed to load datasets from disk: {e}") from e

    if not iid and dirichlet_alpha <= 0:
        raise ValueError("For non-IID data, dirichlet_alpha must be greater than 0.")

    if iid:
        split_dataset = split_dataset.shuffle(seed=seed)  # shuffle dataset to ensure reproducibility
        partitioner = IidPartitioner(
            num_partitions=num_partitions,
        )
        print("[INFO] Using IID data partitioning.")
    else:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=dirichlet_alpha,
            min_partition_size=0,
            seed=seed
        )
        print("[INFO] Using non-IID Dirichlet data partitioning.")

    partitioner.dataset = split_dataset
    client_dataset = partitioner.load_partition(partition_id=partition_id)

    data_split = client_dataset.train_test_split(test_size=0.2, seed=seed)

    transform = get_transforms_custom()
    trainset = HFDatasetToTorch(data_split["train"], transform=transform)
    testset = HFDatasetToTorch(data_split["test"], transform=transform)

    g = torch.Generator()
    g.manual_seed(seed)
    ctx = mp.get_context("spawn")  # avoids the default fork() -> warning/deadlock risk

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g,
                             num_workers=2, multiprocessing_context=ctx, )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, multiprocessing_context=ctx, )

    print(f"[Client {partition_id}] Loaded {len(trainset)} train samples, {len(testset)} test samples.")
    return trainloader, testloader


def ensure_split_data_exists():
    if not all(Path(path).exists() for path in [D1_SPLIT_PATH, D2_SPLIT_PATH, D3_SPLIT_PATH, D4_SPLIT_PATH]):
        split_cifar10_for_target_and_shadow()
