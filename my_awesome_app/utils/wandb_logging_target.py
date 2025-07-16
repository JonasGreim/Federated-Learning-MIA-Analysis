import os
from flwr_datasets.visualization import plot_label_distributions

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def visualize_label_distribution(partitioner, output_dir: str):
    """Save bar and heatmap plots of label distribution per partition."""
    os.makedirs(output_dir, exist_ok=True)

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

    fig_bar.savefig(os.path.join(ROOT_DIR, output_dir, "bar_label_distribution.png"), dpi=300)
    fig_heatmap.savefig(os.path.join(ROOT_DIR, output_dir, "heatmap_label_distribution.png"), dpi=300)
