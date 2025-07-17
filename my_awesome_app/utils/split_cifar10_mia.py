from torchvision.datasets import CIFAR10
import os
import numpy as np
from collections import defaultdict
from datasets import load_dataset, Dataset, DatasetDict


# Function to create CIFAR-10 splits for target model and MIA using Hugging Face datasets
def create_split_files_hf(
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
    dataset = load_dataset("cifar10", split="train")  # 50,000 examples
    targets = dataset["label"]

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
        ds_split = dataset.select(indices)
        ds_split.save_to_disk(os.path.join(split_dir, name))
        print(f"{name}: {len(ds_split)} samples")

    print("✅ Hugging Face CIFAR-10 splits created and saved.")
    # D1: train set for target model (10000 samples, 1000 per class)
    # D2: test set for target model (10000 samples, 1000 per class)
    # D3: train set for shadow model (15000 samples, 1500 per class)
    # D4: test set for shadow model (15000 samples, 1500 per class)

