import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import json
from experiments_data_analysis.file_name_settings import METRICS_ANALYSIS

matplotlib.use("Agg")


def log_per_class_metrics2(run_folder: Path, images_folder: Path, class_names: list = None):
    with open(run_folder / "mia_per_class.json", "r") as f:
        per_class_metrics = json.load(f)

    metrics = list(next(iter(per_class_metrics.values())).keys())

    class_keys = sorted(per_class_metrics.keys())
    if class_names is None:
        class_names = [str(cls) for cls in class_keys]  # fallback if not passed

    for metric in metrics:
        values = [per_class_metrics[cls].get(metric, 0.0) for cls in class_keys]

        plt.figure(figsize=(8, 6))
        plt.bar(class_names, values, color='#1f77b4', edgecolor='black', alpha=1.0)
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xlabel("Klasse")
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.ylim(0, 1.0)
        plt.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.3)
        plt.tight_layout()
        plt.savefig(images_folder / f"per_class_{metric}_metrics2.png", dpi=300, bbox_inches="tight")
        plt.close()


# path = METRICS_ANALYSIS / "side_experiments_clients" / "mia" / "run36-model_ckp:0" / "images"
# run_folder2 = METRICS_ANALYSIS / "side_experiments_clients" / "mia" / "run36-model_ckp:0"
# class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#
# log_per_class_metrics2(run_folder=run_folder2, images_folder=path, class_names=class_names)


def log_overall_metrics_with_error_bars2(
        run_folder: Path,
        images_folder: Path
):
    with open(run_folder / "mia_summary.json", "r") as f:
        metrics_summary = json.load(f)

    # Extract names, means, and stds from the dict
    metrics_names = list(metrics_summary.keys())
    means = [metrics_summary[m]["mean"] for m in metrics_names]
    stds = [metrics_summary[m]["std"] for m in metrics_names]

    # Create the plot
    plt.figure(figsize=(8, 6))
    bars = plt.bar(
        metrics_names, means,
        yerr=stds, capsize=6,
        color='#1f77b4', edgecolor='black', alpha=1.0
    )

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.ylabel("Metrikwert")
    plt.xlabel("Angriffsmetrik")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.3)

    # Add numeric labels
    for bar, mean, std in zip(bars, means, stds):
        y_pos = mean + std
        text_y_pos = y_pos + 0.03
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            text_y_pos,
            f"{mean:.2f}±{std:.2f}",
            ha='center', va='bottom', fontsize=9
        )

    plt.savefig(images_folder / "overall_metrics_with_error_bars2.png", dpi=300)
    plt.close()

# path2 = METRICS_ANALYSIS / "side_experiments_clients" / "mia" / "run36-model_ckp:0" / "images"
# mia_summary_metrics_path = METRICS_ANALYSIS / "side_experiments_clients" / "mia" / "run36-model_ckp:0"
#
# log_overall_metrics_with_error_bars2(
#     run_folder=mia_summary_metrics_path,
#     images_folder=path2
# )
