import pandas as pd
from experiments_data_analysis.data_interface.column_names_merged_table import Col
from pathlib import Path
from experiments_data_analysis.create_figures.main_experiments.create_figures_utils import create_analysis_figures_correlation, create_heatmap


def create_analysis_figures_corr(
        data_path: Path,
        figures_dir: Path
):
    # Load merged data
    merged = pd.read_csv(data_path)


    #  --- 1 Heatmap Correlation Matrix
    cols_names = [
        Col.METRICS_AUC,
        Col.OVERFITTING_GAP_LOSS,
        Col.OVERFITTING_GAP_ACC,
        Col.OVERFITTING_GAP_LOSS_CLIENTS,
        Col.OVERFITTING_GAP_ACC_CLIENTS
    ]

    labels = [
        "MIA-AUC",
        "OGL (Server)",
        "OGA (Server)",
        "OGL (Clients)",
        "OGA (Clients)"
    ]

    create_heatmap(dataset=merged, col_names=cols_names, labels=labels, figures_dir=figures_dir)

    #  --- 2 Pairplot Overfitting Gap Loss vs MIA-AUC
    create_analysis_figures_correlation(dataset=merged, x_value=Col.METRICS_AUC,
                                        x_label="MIA-AUC", y_value=Col.OVERFITTING_GAP_LOSS,
                                        y_label="Overfitting Gap Loss", figures_dir=figures_dir)

    #  --- 3 Pairplot Overfitting Gap Accuracy vs MIA AUC
    create_analysis_figures_correlation(dataset=merged, x_value=Col.METRICS_AUC,
                                        x_label="MIA-AUC", y_value=Col.OVERFITTING_GAP_ACC_CLIENTS,
                                        y_label="Overfitting Gap Acc. (Clients)", figures_dir=figures_dir)

    print("✅ Correlation figures created successfully.")
