import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from experiments_data_analysis.data_interface.column_names_merged_table import Col
from pathlib import Path
import locale


def plot_parameter_per_model(dataset: pd.DataFrame, x_value: Col, x_label: str, figures_dir: Path,
                             order: list[str] = None) -> None:
    g = sns.catplot(
        data=dataset,
        x=x_value,
        y=Col.METRICS_AUC,
        col=Col.MODEL,
        kind="box",
        sharey=True,
        height=4,
        aspect=1,
        order=order,
        showfliers=False,
    )
    g.set_axis_labels(x_var=x_label, y_var="MIA-AUC")
    new_titles = {
        "Shokri-CNN": "Shokri-CNN",
        "ResNet-18": "ResNet-18",
    }

    for ax, model_name in zip(g.axes.flat, g.col_names):
        nice_title = new_titles.get(model_name, model_name)
        ax.set_title(nice_title)
        ax.grid(axis="y")

    g.figure.savefig(figures_dir / f"per_model_{x_value.value}.png", dpi=300)
    plt.close()


def create_analysis_figures_both_models(dataset: pd.DataFrame, x_value: Col, x_label: str, figures_dir: Path,
                                        order: list[str] = None) -> None:
    plt.figure()
    sns.boxplot(data=dataset, x=x_value, y=Col.METRICS_AUC, order=order)
    plt.xlabel(x_label)
    plt.ylabel("MIA-AUC")
    ax = plt.gca()
    ax.grid(axis="y")
    plt.tight_layout()
    plt.savefig(figures_dir / f"both_models_effect_{x_value.value}.png", dpi=300)
    plt.close()


def create_analysis_figures_correlation(dataset: pd.DataFrame, x_value: Col, x_label: str, y_value: Col, y_label: str,
                                        figures_dir: Path) -> None:
    g = sns.pairplot(
        dataset,
        vars=[
            x_value,
            y_value,
        ],
        diag_kind="kde"
    )

    axes = g.axes  # rename axis
    axes[0, 0].grid(axis="both")
    axes[0, 1].grid(axis="both")
    axes[1, 0].grid(axis="both")
    axes[1, 1].grid(axis="both")

    axes[0, 0].set_ylabel(x_label)
    axes[0, 1].set_ylabel(x_label)
    axes[1, 0].set_xlabel(x_label)
    axes[1, 0].set_ylabel(y_label)
    axes[1, 1].set_xlabel(y_label)

    g.figure.tight_layout()
    g.figure.subplots_adjust(right=1.2)

    y_label = y_label.replace(" ", "_").replace(".", "")
    g.savefig(figures_dir / f"pairplot_overfitting_auc_{y_label}.png", dpi=300)
    plt.close()


def create_heatmap(dataset: pd.DataFrame, col_names: list[Col], labels: list[str], figures_dir: Path) -> None:
    corr = dataset[col_names].corr()
    plt.figure()
    ax = sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2n")
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels, rotation=0, va="center")
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(figures_dir / "heatmap_overfitting_auc.png", dpi=300, bbox_inches="tight")
    plt.close()
