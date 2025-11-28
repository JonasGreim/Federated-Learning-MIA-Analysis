import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import json
from experiments_data_analysis.file_name_settings import METRICS_ANALYSIS

matplotlib.use("Agg")

def log_per_class_metrics(per_class_metrics: dict,  images_folder: Path, class_names: list = None):

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'FAR']

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


path = METRICS_ANALYSIS / "side_experiments_clients" / "mia" / "run36-model_ckp:0" / "images"
per_class_metrics_path = METRICS_ANALYSIS / "side_experiments_clients" / "mia" / "run36-model_ckp:0" / "mia_per_class.json"
with open(per_class_metrics_path, "r") as f:
    per_class_metrics = json.load(f)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

log_per_class_metrics(per_class_metrics=per_class_metrics, images_folder=path, class_names=class_names)


def log_overall_metrics_with_error_bars(
        metrics_dict: dict,
        images_folder: Path
):
    # Extract names, means, and stds from the dict
    metrics_names = list(metrics_dict.keys())
    means = [metrics_dict[m]["mean"] for m in metrics_names]
    stds = [metrics_dict[m]["std"] for m in metrics_names]

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

path2 = METRICS_ANALYSIS / "side_experiments_clients" / "mia" / "run36-model_ckp:0" / "images"
mia_summary_metrics_path = METRICS_ANALYSIS / "side_experiments_clients" / "mia" / "run36-model_ckp:0" / "mia_summary.json"
with open(mia_summary_metrics_path, "r") as f:
    metrics = json.load(f)

log_overall_metrics_with_error_bars(
    metrics,
    images_folder=path2
)