from flwr_datasets.visualization import plot_label_distributions
from path_settings import METRICS_DIR, ensure_dir_exist, D3_SPLIT_PATH, D1_SPLIT_PATH, ROOT_DIR
from pathlib import Path
import wandb
from datetime import datetime, UTC
from datasets import load_from_disk
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
import matplotlib
import warnings

# all wandb logging is executed from the server side (server collects all metrics from clients)
matplotlib.use("Agg")


def run_data_partitioning_for_visualization(iid: bool, dirichlet_alpha: float, num_partitions: int,
                                            train_target_model_as_shadow_model: bool, metric_save_folder: Path,
                                            seed: int) -> None:
    if train_target_model_as_shadow_model:
        split = D3_SPLIT_PATH
    else:
        split = D1_SPLIT_PATH

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

    visualize_label_distribution(partitioner, metric_save_folder)


def visualize_label_distribution(partitioner, output_dir: Path = METRICS_DIR):
    """Save bar and heatmap plots of label distribution per partition."""
    ensure_dir_exist(output_dir)

    fig_bar, ax, _ = plot_label_distributions(
        partitioner,
        label_name="label",
        plot_type="bar",
        size_unit="absolute",
        partition_id_axis="x",
        verbose_labels=True,
        title="Verteilung der Klassen pro Client",
    )
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel("Client")
    ax.set_ylabel("Anzahl der Stichproben")
    ax.legend(
        title="Klassen",
        bbox_to_anchor=(1.02, 1.0),  # x>1 puts it outside to the right
        loc="upper left",
        frameon=False,
        ncol=1,
        borderaxespad=0.0,
    )
    fig_bar.subplots_adjust(right=0.70)


    fig_heatmap, ax_hm, _ = plot_label_distributions(
        partitioner,
        label_name="label",
        plot_type="heatmap",
        size_unit="absolute",
        partition_id_axis="x",
        legend=True,
        verbose_labels=True,
        title="Verteilung der Klassen pro Client",
        plot_kwargs={
            "annot": True,
        },
    )
    cbar = fig_heatmap.get_axes()[1]
    cbar.set_ylabel("Anzahl der Stichproben", rotation=270, labelpad=15)
    ax_hm.set_xlabel("Client")
    ax_hm.set_ylabel("Klasse")
    fig_heatmap.tight_layout()

    fig_bar.savefig(output_dir / "bar_label_distribution.png", dpi=300)
    fig_heatmap.savefig(output_dir / "heatmap_label_distribution.png", dpi=300)
    wandb.log({
        "Label Distribution Bar": wandb.Image(fig_bar),
        "Label Distribution Heatmap": wandb.Image(fig_heatmap)
    }, step=0)


def initialize_wandb_run(project_name: str, run_name: str, config: dict = None, directory: Path = ROOT_DIR):
    if not wandb.run:
        disable_unnecessary_warnings()
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M")
        wandb.init(project=project_name, name=f"{run_name}-{timestamp}", config=config, dir=directory, settings=wandb.Settings(init_timeout=240, start_method="thread"))


def wandb_log_metrics(metrics: dict, step: int):
    wandb.log(metrics, step=step)


def wandb_upload_artifact_model(artifact_name: str, artifact_path: Path, wandb_type: str = "model"):
    """Upload an artifact to the current wandb run."""
    artifact = wandb.Artifact(name=artifact_name, type=wandb_type)
    artifact.add_file(str(artifact_path))
    wandb.log_artifact(artifact)
    print(f"Artifact {artifact_name} uploaded to wandb successfully.", flush=True)


def disable_unnecessary_warnings():
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module=r"^wandb\.analytics\.sentry$",
        message=r".*Scope\.user.*deprecated.*Scope\.set_user\(\).*",
    )
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module=r"^google\.protobuf\.internal\.well_known_types$",
        message=r".*datetime\.datetime\.utcnow\(\).*",
    )

def wandb_save_file(path: Path):
    wandb.save(str(path))
