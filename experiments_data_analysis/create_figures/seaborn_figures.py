import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from experiments_data_analysis.data_interface.column_names import Col
from experiments_data_analysis.utils.merge_utils import map_regularization_diagram
from pathlib import Path


def create_analysis_figures(
        data_path: Path,
        figures_dir: Path
):
    # Load data
    merged = pd.read_csv(data_path)

    # --- 1 Effect of Local Epochs ---
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=merged, x=Col.LOCAL_EPOCHS, y=Col.METRICS_AUC)
    plt.title("Einfluss der lokaler Epochen auf die MIA AUC")
    plt.xlabel("Lokale Epochen")
    plt.ylabel("MIA AUC")
    plt.tight_layout()
    plt.savefig(figures_dir / "effect_local_epochs.png", dpi=300)
    plt.close()

    # --- 2 Architecture Influence ---
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=merged, x=Col.MODEL, y=Col.METRICS_AUC)
    plt.title("Architektonischer Einfluss auf die MIA AUC")
    plt.xlabel("Modellarchitektur")
    plt.ylabel("MIA AUC")
    plt.tight_layout()
    plt.savefig(figures_dir / "architecture_influence.png", dpi=300)
    plt.close()

    # --- 3 Data Distribution Influence ---
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=merged, x=Col.DATA_DISTRIBUTION, y=Col.METRICS_AUC, order=["IID", "semi-non-IID", "non-IID"])
    plt.title("Einfluss der Datenverteilung auf die MIA AUC")
    plt.xlabel("Datenverteilung")
    plt.ylabel("MIA AUC")
    plt.tight_layout()
    plt.savefig(figures_dir / "data_distribution_influence.png", dpi=300)
    plt.close()

    # --- 4 Interaction: Architecture × Data Distribution ---
    g = sns.catplot(
        data=merged,
        x=Col.DATA_DISTRIBUTION,
        y=Col.METRICS_AUC,
        hue=Col.MODEL,
        kind="bar",
        height=5, aspect=1.2
    )
    g.figure.suptitle("MIA AUC: Architektur × Datenverteilung", y=1.02)
    g.set_axis_labels(Col.DATA_DISTRIBUTION, Col.METRICS_AUC)
    g.savefig(figures_dir / "interaction_architecture_data.png", dpi=300)
    plt.close()

    # --- 5 Regularization Influence ---
    merged[Col.REGULARIZATION] = merged[Col.REGULARIZATION].apply(map_regularization_diagram)
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=merged, x=Col.REGULARIZATION, y=Col.METRICS_AUC)
    plt.title("Einfluss von Regularisierung auf die MIA AUC")
    plt.xlabel("Regularisierung")
    plt.ylabel("MIA AUC")
    plt.tight_layout()
    plt.savefig(figures_dir / "regularization_influence.png", dpi=300)
    plt.close()

    # --- 6 Regularization per Model Architecture ---
    plt.figure(figsize=(10, 4))
    g = sns.catplot(
        data=merged,
        x=Col.REGULARIZATION,
        y=Col.METRICS_AUC,
        col=Col.MODEL,
        kind="box",
        sharey=True,
        height=4,
        aspect=1
    )
    g.set_axis_labels("Regularisierung", "MIA AUC")
    new_titles = {
        "Shokri-CNN": "Shokri CNN",
        "ResNet-18": "ResNet-18",
        # add more mappings if needed
    }

    for ax, model_name in zip(g.axes.flat, g.col_names):
        nice_title = new_titles.get(model_name, model_name)
        ax.set_title(nice_title)
    g.figure.suptitle("Einfluss der Regularisierung getrennt nach Modellarchitektur", y=1.05)
    plt.savefig(figures_dir / "regularization_per_model.png", dpi=300)
    plt.close()
