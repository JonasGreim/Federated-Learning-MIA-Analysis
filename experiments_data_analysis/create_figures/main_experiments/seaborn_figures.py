import pandas as pd
from experiments_data_analysis.data_interface.column_names_merged_table import Col
from experiments_data_analysis.utils.create_figures_main import plot_parameter_per_model, \
    create_analysis_figures_both_models
from experiments_data_analysis.utils.merge_utils import map_regularization_diagram
from pathlib import Path


def create_analysis_figures(
        data_path: Path,
        figures_dir: Path
):
    # Load merged data
    merged = pd.read_csv(data_path)

    # --- 1 Effect of Local Epochs ---
    create_analysis_figures_both_models(dataset=merged, x_value=Col.LOCAL_EPOCHS, x_label="Lokale Epochen",
                                        figures_dir=figures_dir)

    # --- 2 Architecture Influence ---
    create_analysis_figures_both_models(dataset=merged, x_value=Col.MODEL, x_label="Modellarchitektur",
                                        figures_dir=figures_dir)

    # --- 3 Data Distribution Influence ---
    create_analysis_figures_both_models(dataset=merged, x_value=Col.DATA_DISTRIBUTION, x_label="Datenverteilung",
                                        figures_dir=figures_dir,
                                        order=["IID", "semi-non-IID", "non-IID"])

    # --- 4 Regularization Influence ---
    merged[Col.REGULARIZATION] = merged[Col.REGULARIZATION].apply(map_regularization_diagram)
    create_analysis_figures_both_models(dataset=merged, x_value=Col.REGULARIZATION, x_label="Regularisierung",
                                        figures_dir=figures_dir)

    # --- 5 Regularization per Model Architecture ---
    plot_parameter_per_model(dataset=merged, x_value=Col.REGULARIZATION, x_label="Regularisierung",
                             figures_dir=figures_dir)
    plot_parameter_per_model(dataset=merged, x_value=Col.LOCAL_EPOCHS, x_label="Lokale Epochen",
                             figures_dir=figures_dir)
    plot_parameter_per_model(dataset=merged, x_value=Col.DATA_DISTRIBUTION, x_label="Datenverteilung",
                             figures_dir=figures_dir, order=["IID", "semi-non-IID", "non-IID"])

    print("✅ Figures created successfully.")
