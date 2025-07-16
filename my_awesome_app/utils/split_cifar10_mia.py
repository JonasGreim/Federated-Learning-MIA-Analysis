from torchvision.datasets import CIFAR10
import os
import numpy as np
from collections import defaultdict
from datasets import load_dataset, Dataset, DatasetDict


def create_split_files_hf() -> None:
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

    # Build D1–D4 splits
    D1, D2, D3, D4 = [], [], [], []
    for cls in range(10):
        D1 += class_indices[cls][:1000]
        D2 += class_indices[cls][1000:2000]
        D3 += class_indices[cls][2000:3500]
        D4 += class_indices[cls][3500:5000]

    # Shuffle each split globally
    for split in [D1, D2, D3, D4]:
        np.random.shuffle(split)

    # Create datasets from indices
    dataset_splits = {
        "D1": dataset.select(D1),
        "D2": dataset.select(D2),
        "D3": dataset.select(D3),
        "D4": dataset.select(D4),
    }

    # Save each split to disk (Arrow format is fast)
    for name, ds in dataset_splits.items():
        ds.save_to_disk(os.path.join(split_dir, f"{name}"))
        print(f"{name}: {len(ds)} samples")

    print("✅ Hugging Face CIFAR-10 splits created and saved.")
    # D1: train set for target model (10000 samples, 1000 per class)
    # D2: test set for target model (10000 samples, 1000 per class)
    # D3: train set for shadow model (15000 samples, 1500 per class)
    # D4: test set for shadow model (15000 samples, 1500 per class)
