from torchvision.datasets import CIFAR10
import os
import numpy as np
from collections import defaultdict
from datasets import load_dataset, Dataset, DatasetDict


# Function to create CIFAR-10 splits for target model and MIA using Hugging Face datasets
# without cifar test set -> only train set is split into target and shadow datasets
def split_cifar10_for_target_and_shadow_without_testset(
        target_train_split_ratio: float,
        target_test_split_ratio: float,
        shadow_train_split_ratio: float,
        shadow_test_split_ratio: float
) -> None:
    total = target_train_split_ratio + target_test_split_ratio + shadow_train_split_ratio + shadow_test_split_ratio
    assert abs(total - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    split_ratios = {
        "D1": target_train_split_ratio,
        "D2": target_test_split_ratio,
        "D3": shadow_train_split_ratio,
        "D4": shadow_test_split_ratio,
    }

    # Define paths
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    split_dir = os.path.join(root_dir, "splits")
    os.makedirs(split_dir, exist_ok=True)

    # Load CIFAR-10 via Hugging Face
    dataset_train = load_dataset("cifar10", split="train")  # 50,000 examples
    targets = dataset_train["label"]

    # Group indices per class
    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)

    # Shuffle each class
    for label in class_indices:
        np.random.shuffle(class_indices[label])

    # Initialize index containers for splits
    split_indices = {name: [] for name in split_ratios}

    # Split per class
    for cls in range(10):
        cls_indices = class_indices[cls]
        total_cls = len(cls_indices)
        start = 0

        for name, ratio in split_ratios.items():
            count = int(ratio * total_cls)
            split_indices[name] += cls_indices[start:start + count]
            start += count

    # Shuffle indices in each split
    for name in split_indices:
        np.random.shuffle(split_indices[name])

    # Select and save datasets
    for name, indices in split_indices.items():
        ds_split = dataset_train.select(indices)
        ds_split.save_to_disk(os.path.join(split_dir, name))
        print(f"{name}: {len(ds_split)} samples")

    print("✅ Hugging Face CIFAR-10 splits created and saved.")


# Function to create CIFAR-10 splits for target model and MIA using Hugging Face datasets
# D1= target train set, D2=target test set, D3=shadow train set, D4=shadow test set
# D2 comes from the original CIFAR-10 test set, so it is not split further
# currently: D1=20.000, D2=10.000, D3=15.000, D4=15.000
def split_cifar10_for_target_and_shadow(
        target_train_ratio: float,
        shadow_train_ratio: float,
        shadow_test_ratio: float
) -> None:
    # The remaining ratio must be ≤ 1.0 since D2 comes from test set
    total_ratio = target_train_ratio + shadow_train_ratio + shadow_test_ratio
    assert abs(total_ratio - 1.0) < 1e-6, "Train split ratios must sum to 1.0 (D2 comes from test set)"

    split_ratios = {
        "D1": target_train_ratio,
        "D3": shadow_train_ratio,
        "D4": shadow_test_ratio,
    }

    split_dir = os.path.join("splits")
    os.makedirs(split_dir, exist_ok=True)

    # Load CIFAR-10 datasets
    dataset_train = load_dataset("cifar10", split="train")  # 50,000
    dataset_test = load_dataset("cifar10", split="test")  # 10,000 — will be used entirely as D2: target test set

    targets = dataset_train["label"]

    # Group indices per class for training data
    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)

    # Shuffle class indices
    for label in class_indices:
        np.random.shuffle(class_indices[label])

    # Allocate training data to D1, D3, D4
    split_indices = {name: [] for name in split_ratios}
    for cls in range(10):
        cls_indices = class_indices[cls]
        total_cls = len(cls_indices)
        start = 0
        for name, ratio in split_ratios.items():
            count = int(ratio * total_cls)
            split_indices[name].extend(cls_indices[start:start + count])
            start += count

    # Shuffle each split
    for name in split_indices:
        np.random.shuffle(split_indices[name])

    # Save D1, D3, D4 from training set
    for name, indices in split_indices.items():
        ds_split = dataset_train.select(indices)
        ds_split.save_to_disk(os.path.join(split_dir, name))
        print(f"{name}: {len(ds_split)} samples")

    # Save D2 directly from test set
    dataset_test.save_to_disk(os.path.join(split_dir, "D2"))
    print(f"D2: {len(dataset_test)} samples (from CIFAR-10 test set)")

    print("✅ CIFAR-10 custom splits created and saved.")


split_cifar10_for_target_and_shadow(
    target_train_ratio=0.4,
    shadow_train_ratio=0.3,
    shadow_test_ratio=0.3)
