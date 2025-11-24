import pandas as pd
from experiments_data_analysis.data_interface.column_names_merged_table import Col
from pathlib import Path


def create_overview_table(
        data_path: Path,
        output_path: Path,
        figures_dir: Path
):
    # Load both CSVs
    full_data_tables = pd.read_csv(data_path)

    keep_columns = [
        Col.MODEL, Col.REGULARIZATION, Col.DATA_DISTRIBUTION, Col.LOCAL_EPOCHS,
        Col.OVERFITTING_GAP_LOSS, Col.OVERFITTING_GAP_ACC, Col.METRICS_AUC
    ]

    overview_table = full_data_tables[keep_columns]

    overview_table = overview_table.rename(columns={
        Col.MODEL: "Model",
        Col.REGULARIZATION: "Reg.",
        Col.DATA_DISTRIBUTION: "Data Dist.",
        Col.LOCAL_EPOCHS: "Local Epochs",
        Col.OVERFITTING_GAP_LOSS: "Overfitting Gap Loss",
        Col.OVERFITTING_GAP_ACC: "Overfitting Gap Acc",
        Col.METRICS_AUC: "MIA AUC",
    })

    # Ensure categorical sorting
    overview_table["Model"] = pd.Categorical(
        overview_table["Model"], categories=["Shokri-CNN", "ResNet-18"], ordered=True
    )
    overview_table["Reg."] = pd.Categorical(
        overview_table["Reg."], categories=[False, True], ordered=True
    )
    overview_table["Data Dist."] = pd.Categorical(
        overview_table["Data Dist."],
        categories=["IID", "semi-non-IID", "non-IID"],
        ordered=True
    )
    overview_table["Local Epochs"] = pd.Categorical(
        overview_table["Local Epochs"], categories=[1, 5, 10], ordered=True
    )

    # Sort in a meaningful grid order
    overview_table = overview_table.sort_values(
        ["Model", "Reg.", "Data Dist.", "Local Epochs"]
    )

    file_name_overview = output_path / "overview.csv"
    overview_table.to_csv(file_name_overview, index=False)

    # to latex format
    latex_table = overview_table.to_latex(index=False, float_format="%.2f")

    file_name_text = figures_dir / "table_latex.tex"
    with open(file_name_text, "w") as f:
        f.write(latex_table)

    print("✅ Overview Tables created successfully.")
