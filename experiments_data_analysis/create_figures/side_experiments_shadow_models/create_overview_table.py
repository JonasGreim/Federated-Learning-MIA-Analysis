import pandas as pd
from experiments_data_analysis.data_interface.column_names_merged_table import Col
from pathlib import Path


def create_overview_table_shadow_experiments(
        data_path: Path,
        output_path: Path,
        figures_dir: Path
):
    # Load both CSVs
    full_data_tables = pd.read_csv(data_path)

    full_data_tables = full_data_tables.rename(columns={
        "model_arch": "model",
    })

    keep_columns = [
        Col.MODEL, Col.NUM_SHADOW_MODELS, Col.TRAIN_SIZE, Col.METRICS_AUC,
    ]

    overview_table = full_data_tables[keep_columns]

    overview_table = overview_table.rename(columns={
        Col.MODEL: "Model",
        Col.NUM_SHADOW_MODELS: "Number of Shadow Models",
        Col.TRAIN_SIZE: "Train Size",
        Col.METRICS_AUC: "MIA AUC",
    })

    # Ensure categorical sorting
    overview_table["Model"] = pd.Categorical(
        overview_table["Model"], categories=["Shokri-CNN", "ResNet-18"], ordered=True
    )

    # Sort in a meaningful grid order
    # overview_table = overview_table.sort_values(
    #     ["Model", "Client Number"]
    # )

    file_name_overview = output_path / "overview.csv"
    overview_table.to_csv(file_name_overview, index=False)

    # to latex format
    latex_table = overview_table.to_latex(index=False, float_format="%.2f")

    file_name_text = figures_dir / "table_latex.tex"
    with open(file_name_text, "w") as f:
        f.write(latex_table)
