import os
import numpy as np
from torchvision.datasets import CIFAR10
from collections import defaultdict

def create_split_files():
    # Define paths
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(root_dir, "data")
    split_dir = os.path.join(root_dir, "splits")

    # Create directories if needed
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)

    # Load CIFAR-10 dataset
    dataset = CIFAR10(root=data_dir, train=True, download=True)
    targets = np.array(dataset.targets)

    # Collect indices per class
    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)

    # Shuffle each class
    for label in class_indices:
        np.random.shuffle(class_indices[label])

    # Build splits
    D1, D2, D3, D4 = [], [], [], []
    for cls in range(10):
        D1 += class_indices[cls][:1000]
        D2 += class_indices[cls][1000:2000]
        D3 += class_indices[cls][2000:3500]
        D4 += class_indices[cls][3500:5000]

    # Shuffle each split globally
    for split in [D1, D2, D3, D4]:
        np.random.shuffle(split)

    # Save to files
    np.save(os.path.join(split_dir, "D1_indices.npy"), np.array(D1))
    np.save(os.path.join(split_dir, "D2_indices.npy"), np.array(D2))
    np.save(os.path.join(split_dir, "D3_indices.npy"), np.array(D3))
    np.save(os.path.join(split_dir, "D4_indices.npy"), np.array(D4))

    print("✅ CIFAR-10 splits created and saved.")

    # Save as .npy
    # D1: train set for target model (10000 samples, 1000 per class)
    # D2: test set for target model (10000 samples, 1000 per class)
    # D3: train set for shadow model (15000 samples, 1500 per class)
    # D4: test set for shadow model (15000 samples, 1500 per class)

