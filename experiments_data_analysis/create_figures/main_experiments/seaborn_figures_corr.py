import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from experiments_data_analysis.data_interface.column_names_merged_table import Col
from pathlib import Path


def create_analysis_figures_corr(
        data_path: Path,
        figures_dir: Path
):
    # Load your merged data
    merged = pd.read_csv(data_path)

    # Select columns of interest
    cols_of_interest = [
        Col.METRICS_AUC,
        Col.OVERFITTING_GAP_LOSS,
        Col.OVERFITTING_GAP_ACC,
        Col.OVERFITTING_GAP_LOSS_CLIENTS,
        Col.OVERFITTING_GAP_ACC_CLIENTS
    ]

    # Compute correlation matrix
    corr = merged[cols_of_interest].corr()

    # Print numeric correlation table
    print(corr.round(2))

    #  --- 1 Plot heatmap
    plt.figure(figsize=(7, 5))
    ax = sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    ax.set_xticklabels([
        "MIA AUC",
        "Overfitting Gap Loss",
        "Overfitting Gap Acc.",
        "Overfitting Gap Loss (Clients)",
        "Overfitting Gap Acc. (Clients)"
    ], rotation=45, ha="right")

    ax.set_yticklabels([
        "MIA AUC",
        "Overfitting Gap Loss",
        "Overfitting Gap Acc.",
        "Overfitting Gap Loss (Clients)",
        "Overfitting Gap Acc. (Clients)"
    ], rotation=0)
    plt.title("Korrelation von Overfitting Gaps und MIA AUC", pad=20)
    plt.tight_layout()
    plt.savefig(figures_dir / "correlation_overfitting_auc.png", dpi=300)
    plt.close()

    #  --- 2 Pairplot Overfitting Gap Loss vs MIA AUC
    g = sns.pairplot(
        merged,
        vars=[
            Col.METRICS_AUC,
            Col.OVERFITTING_GAP_LOSS
        ],
        diag_kind="kde"
    )
    axes = g.axes  # rename axis
    axes[0, 0].set_ylabel("MIA AUC")
    axes[0, 1].set_ylabel("MIA AUC")
    axes[1, 0].set_xlabel("MIA AUC")
    axes[1, 0].set_ylabel("Overfitting Gap Loss")
    axes[1, 1].set_xlabel("Overfitting Gap Loss")

    g.figure.suptitle("Korrelation von Overfitting Gap Loss und MIA AUC", y=1.03)
    g.figure.subplots_adjust(top=0.95)  # increase or decrease top margin
    g.savefig(figures_dir / "pairplot_overfitting_auc.png", dpi=300)
    plt.close()

    #  --- 3 Pairplot Overfitting Gap Accuracy vs MIA AUC
    g = sns.pairplot(
        merged,
        vars=[
            Col.METRICS_AUC,
            Col.OVERFITTING_GAP_ACC_CLIENTS,
        ],
        diag_kind="kde"
    )
    axes = g.axes  # rename axis
    axes[0, 0].set_ylabel("MIA AUC")
    axes[0, 1].set_ylabel("MIA AUC")
    axes[1, 0].set_xlabel("MIA AUC")
    axes[1, 0].set_ylabel("Overfitting Gap Acc.")
    axes[1, 1].set_xlabel("Overfitting Gap Acc.")
    g.figure.suptitle("Korrelation von Overfitting Gap Accuracy und MIA AUC", y=1.03)
    g.figure.subplots_adjust(top=0.95)  # increase or decrease top margin
    g.savefig(figures_dir / "pairplot_overfitting_auc_acc.png", dpi=300)
    plt.close()
