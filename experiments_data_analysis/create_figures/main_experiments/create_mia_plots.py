import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from path_settings import METRICS_DIR_MIA

matplotlib.use("Agg")

def log_per_class_metrics(per_class_metrics: dict, class_names: list = None,
                          metric_save_folder: Path = METRICS_DIR_MIA):
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
        plt.savefig(metric_save_folder / f"per_class_{metric}_metrics.png", dpi=300, bbox_inches="tight")
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
    plt.close()
