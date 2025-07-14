from collections import Counter
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import wandb


def log_class_distribution(dataset: Subset, wandb_cluster_name: str = "dataset_distribution", wandb_plot_prefix: str = "split") -> None:
    # Extract class labels from subset
    targets = [dataset.dataset.targets[i] for i in dataset.indices]  # creates a list of class labels for all samples in the subset
    class_counts = Counter(targets)  # Dictionary with class labels as keys and counts as values

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    counts = [class_counts.get(i, 0) for i in range(10)]

    # Log class distribution as a WandB table
    table_data = [[class_names[i], counts[i]] for i in range(10)]
    table = wandb.Table(data=table_data, columns=["class", "count"])
    wandb.log({f"{wandb_cluster_name}/{wandb_plot_prefix}/class_distribution": wandb.plot.bar(
        table, "class", "count", title=f"{wandb_plot_prefix.capitalize()}"
    )})

    # Log histogram-style bar chart for class distribution
    # plt.figure(figsize=(8, 6))
    # plt.bar(class_names, counts, alpha=0.7, color='skyblue', edgecolor='black')
    # plt.xlabel("CIFAR-10 Class")
    # plt.ylabel("Number of Samples")
    # plt.title(f"Histogram of {wandb_cluster_name.capitalize()} Class Distribution")
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # wandb.log({f"{wandb_cluster_name}/{wandb_plot_prefix}_class_histogram": wandb.Image(plt)})
    # plt.close()
    
    
    