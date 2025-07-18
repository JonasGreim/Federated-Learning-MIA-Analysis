from flwr_datasets.visualization import plot_label_distributions
from path_settings import METRICS_DIR, ensure_dir_exist
from pathlib import Path


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
