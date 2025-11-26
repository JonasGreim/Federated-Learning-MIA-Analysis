from datasets import Dataset
from collections import Counter
import matplotlib.pyplot as plt
import wandb
import matplotlib
from pathlib import Path
from flower.utils.wandb_logging import wandb_save_file
from path_settings import METRICS_DIR_MIA
import json

matplotlib.use("Agg")


def log_class_distribution(
        hf_dataset: Dataset,
        class_names: list = None,
        wandb_cluster_name: str = "dataset_distribution",
        wandb_plot_prefix: str = "split",
        metric_save_folder: Path = METRICS_DIR_MIA,
) -> None:
    try:
        labels = hf_dataset["label"]
    except Exception as e:
        print(f"[log_class_distribution_hf] Failed to access column label: {e}")
        return

    class_counts = Counter(labels)

    # Determine number of classes
    n_classes = max(class_counts.keys()) + 1 if class_counts else 0

    # Use class indices if names not provided
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    counts = [class_counts.get(i, 0) for i in range(len(class_names))]

    # Log class distribution as a WandB table
    table_data = [[class_names[i], counts[i]] for i in range(len(class_names))]
    table = wandb.Table(data=table_data, columns=["class", "count"])
    wandb.log({f"{wandb_cluster_name}/{wandb_plot_prefix}/class_distribution": wandb.plot.bar(
        table, "class", "count", title=f"Klassenhäufigkeit des {wandb_plot_prefix}"
    )})

    # Log histogram-style bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(class_names, counts, color='#1f77b4', edgecolor='black', alpha=1.0)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel("Klasse")
    plt.ylabel("Anzahl der Stichproben")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.3)
    plt.tight_layout()
    metric_save_file = metric_save_folder / f"histogram_class_distribution_train_{wandb_plot_prefix}.png"
    plt.savefig(metric_save_file, dpi=300, bbox_inches="tight")
    wandb.log({f"{wandb_cluster_name}/{wandb_plot_prefix}_class_histogram": wandb.Image(plt)})
    plt.close()


def log_per_class_metrics(per_class_metrics: dict, class_names: list = None,
                          metric_save_folder: Path = METRICS_DIR_MIA):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'FAR']

    class_keys = sorted(per_class_metrics.keys())
    if class_names is None:
        class_names = [str(cls) for cls in class_keys]  # fallback if not passed

    # Save per-class metrics to JSON
    json_data_path = metric_save_folder / "mia_per_class.json"
    with open(json_data_path, "w") as f:
        json.dump(per_class_metrics, f, indent=4)

    wandb_save_file(json_data_path)

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

        plt.savefig(metric_save_folder / f"per_class_{metric}_metrics.png", dpi=300, bbox_inches="tight")
        wandb.log({f"attack_eval/per_class_{metric}_vertical": wandb.Image(plt)})
        plt.close()


def log_overall_metrics_with_error_bars(
        accuracy, std_accuracy,
        precision, std_precision,
        recall, std_recall,
        f1, std_f1,
        auc, std_auc,
        metric_save_folder: Path = METRICS_DIR_MIA
):
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    means = [accuracy, precision, recall, f1, auc]
    stds = [std_accuracy, std_precision, std_recall, std_f1, std_auc]

    for name, value in zip(metrics_names, means):
        wandb.run.summary[f"metrics/{name}"] = value

    for name, value in zip(metrics_names, stds):
        wandb.run.summary[f"std/std_{name}"] = value

    # Save overall metrics to JSON
    metrics_payload = {
        name: {"mean": float(m), "std": float(s)}
        for name, m, s in zip(metrics_names, means, stds)
    }

    json_data_path = metric_save_folder / "mia_summary.json"
    with open(json_data_path, "w") as f:
        json.dump(metrics_payload, f, indent=4)

    wandb_save_file(json_data_path)

    # Create the plot
    plt.figure(figsize=(8, 6))
    bars = plt.bar(metrics_names, means, yerr=stds, capsize=6, color='#1f77b4', edgecolor='black', alpha=1.0)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.ylabel("Metrikwert")
    plt.xlabel("Angriffsmetrik")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.3)

    # Add numeric labels slightly above the error bars
    for bar, mean, std in zip(bars, means, stds):
        # Calculate the top of the error bar (mean + std)
        y_pos = mean + std
        # Add a small buffer to the y-position for the text
        text_y_pos = y_pos + 0.03

        plt.text(bar.get_x() + bar.get_width() / 2.0, text_y_pos,
                 f"{mean:.2f}±{std:.2f}", ha='center', va='bottom', fontsize=9)

    plt.savefig(metric_save_folder / "overall_metrics_with_error_bars.png", dpi=300)
    # Log the plot to wandb and close it
    wandb.log({"attack_eval/overall_metrics_with_error_bars": wandb.Image(plt)})
    plt.close()
