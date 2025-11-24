import pandas as pd
from experiments_data_analysis.data_interface.column_names_merged_table import Col
from pathlib import Path


def create_overview_table_clients_experiments(
        data_path: Path,
        output_path: Path,
        figures_dir: Path
):
    # Load both CSVs
    full_data_tables = pd.read_csv(data_path)

    keep_columns = [
        Col.MODEL, Col.NUM_CLIENTS, Col.METRICS_AUC,
        Col.OVERFITTING_GAP_LOSS, Col.OVERFITTING_GAP_ACC
    ]

    overview_table = full_data_tables[keep_columns]

    overview_table = overview_table.rename(columns={
        Col.MODEL: "Model",
        Col.NUM_CLIENTS: "Client Number",
        Col.METRICS_AUC: "MIA AUC",
        Col.OVERFITTING_GAP_LOSS: "Overfitting Gap Loss",
        Col.OVERFITTING_GAP_ACC: "Overfitting Gap Acc",
    })

    # Ensure categorical sorting
    overview_table["Model"] = pd.Categorical(
        overview_table["Model"], categories=["Shokri-CNN", "ResNet-18"], ordered=True
    )

    overview_table["Client Number"] = pd.Categorical(
        overview_table["Client Number"], categories=[2, 5, 10], ordered=True
    )

    # Sort in a meaningful grid order
    overview_table = overview_table.sort_values(
        ["Model", "Client Number"]
    )

    file_name_overview = output_path / "overview.csv"
    overview_table.to_csv(file_name_overview, index=False)

    # to latex format
    latex_table = overview_table.to_latex(index=False, float_format="%.2f")

    file_name_text = figures_dir / "table_latex.tex"
    with open(file_name_text, "w") as f:
        f.write(latex_table)

    print("✅ Overview table (client number) created successfully.")
