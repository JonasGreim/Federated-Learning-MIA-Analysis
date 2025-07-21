from flwr_datasets.visualization import plot_label_distributions
from path_settings import METRICS_DIR, ensure_dir_exist
from pathlib import Path
import wandb
from datetime import datetime
from datasets import load_from_disk
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
import matplotlib


def run_data_partitioning_for_visualization(iid: bool, dirichlet_alpha: float, num_partitions: int, split: Path,
                                            seed: int) -> None:
    try:
        split_dataset = load_from_disk(split)
    except Exception as e:
        raise RuntimeError(f"Target model: Failed to load datasets from disk: {e}") from e

    if iid:
        split_dataset = split_dataset.shuffle(seed=seed)
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
            seed=seed,
        )
        print("[INFO] Using non-IID Dirichlet data partitioning.")

    partitioner.dataset = split_dataset

    matplotlib.use("Agg")
    visualize_label_distribution(partitioner, METRICS_DIR)


def visualize_label_distribution(partitioner, output_dir: Path = METRICS_DIR):
    """Save bar and heatmap plots of label distribution per partition."""
    ensure_dir_exist(output_dir)

    fig_bar, _, _ = plot_label_distributions(
        partitioner,
        label_name="label",
        plot_type="bar",
        size_unit="absolute",
        partition_id_axis="x",
        legend=True,
        verbose_labels=True,
        title="Per Partition Labels Distribution",
    )

    fig_heatmap, _, _ = plot_label_distributions(
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

    fig_bar.savefig(output_dir / "bar_label_distribution.png", dpi=300)
    fig_heatmap.savefig(output_dir / "heatmap_label_distribution.png", dpi=300)
    wandb.log({
        "Label Distribution Bar": wandb.Image(fig_bar),
        "Label Distribution Heatmap": wandb.Image(fig_heatmap)
    })


def initialize_wandb_run(project_name: str, run_name: str):
    if not wandb.run:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wandb.init(project=project_name, name=f"{run_name}-{timestamp}")


def wandb_log_metrics(metrics: dict, step: int):
    wandb.log(metrics, step=step)
