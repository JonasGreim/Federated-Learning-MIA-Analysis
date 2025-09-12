from datasets import Dataset
from collections import Counter
import matplotlib.pyplot as plt
import wandb
import matplotlib

matplotlib.use("Agg")


def log_class_distribution(
        hf_dataset: Dataset,
        class_names: list = None,
        wandb_cluster_name: str = "dataset_distribution",
        wandb_plot_prefix: str = "split"
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
        table, "class", "count", title=f"{wandb_plot_prefix.capitalize()} Class Distribution"
    )})

    # Log histogram-style bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(class_names, counts, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.title(f"Histogram of {wandb_cluster_name.capitalize()} Class Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    wandb.log({f"{wandb_cluster_name}/{wandb_plot_prefix}_class_histogram": wandb.Image(plt)})
    plt.close()


def log_per_class_metrics(per_class_metrics: dict, class_names: list = None):
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc', 'far']

    class_keys = sorted(per_class_metrics.keys())
    if class_names is None:
        class_names = [str(cls) for cls in class_keys]  # fallback if not passed

    for metric in metrics:
        values = [per_class_metrics[cls].get(metric, 0.0) for cls in class_keys]

        plt.figure(figsize=(10, 6))
        plt.bar(class_names, values, edgecolor='black')
        plt.xlabel("Class")
        plt.ylabel(metric.capitalize())
        plt.title(f"Per-Class {metric.replace('_', ' ').capitalize()} of Attack Model")
        plt.ylim(0, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        wandb.log({f"attack_eval/per_class_{metric}_vertical": wandb.Image(plt)})
        plt.close()


def log_overall_metrics_with_error_bars(
        accuracy, std_accuracy,
        precision, std_precision,
        recall, std_recall,
        f1, std_f1,
        auc, std_auc,
        far, std_far
):
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'FAR']
    means = [accuracy, precision, recall, f1, auc, far]
    stds = [std_accuracy, std_precision, std_recall, std_f1, std_auc, std_far]

    for name, value in zip(metrics_names, means):
        wandb.run.summary[f"metrics/{name}"] = value

    for name, value in zip(metrics_names, stds):
        wandb.run.summary[f"std/std_{name}"] = value

    plt.figure(figsize=(8, 5))
    bars = plt.bar(metrics_names, means, yerr=stds, capsize=6, color='skyblue', edgecolor='black')
    plt.ylabel("Score")
    plt.title("Overall Attack Metrics with Std Deviation")
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Add numeric labels on top of bars
    for bar, mean, std in zip(bars, means, stds):
        plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.02,
                 f"{mean:.2f}±{std:.2f}", ha='center', va='bottom', fontsize=9)

    wandb.log({"attack_eval/overall_metrics_with_error_bars": wandb.Image(plt)})
    plt.close()
