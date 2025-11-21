import pandas as pd
from experiments_data_analysis.data_interface.column_names_merged_table import Col
from pathlib import Path
from experiments_data_analysis.utils.merge_utils import map_model_name
from experiments_data_analysis.utils.shadow_models import map_run_name


def create_overview_table_shadow_experiments(
        data_path: Path,
        output_path: Path,
        figures_dir: Path
):
    # Load both CSVs
    full_data_tables = pd.read_csv(data_path)

    # Rename columns to match interface
    full_data_tables = full_data_tables.rename(columns={
        "model_arch": Col.MODEL,
        "run_name": Col.RUN_NAME,
    })

    keep_columns = [
        Col.RUN_NAME, Col.MODEL, Col.NUM_SHADOW_MODELS, Col.TRAIN_SIZE, Col.METRICS_AUC,
    ]

    overview_table = full_data_tables[keep_columns].copy()

    # Map model names for better readability
    overview_table[Col.MODEL] = overview_table[Col.MODEL].apply(map_model_name)
    overview_table[Col.DATA_DISTRIBUTION] = overview_table[Col.RUN_NAME].apply(map_run_name)

    # Ensure categorical sorting
    overview_table = overview_table.sort_values(
        by=[Col.MODEL, Col.DATA_DISTRIBUTION, Col.NUM_SHADOW_MODELS],
        ascending=[True, True, True]
    )

    # overview_table = overview_table.rename(columns={
    #     Col.RUN_NAME: "run-name",
    #     Col.MODEL: "Model",
    #     Col.NUM_SHADOW_MODELS: "Number of Shadow Models",
    #     Col.TRAIN_SIZE: "Train Size",
    #     Col.METRICS_AUC: "MIA AUC",
    # })

    file_name_overview = output_path / "overview.csv"
    overview_table.to_csv(file_name_overview, index=False)

    # to latex format
    latex_table = overview_table.to_latex(index=False, float_format="%.2f")

    file_name_text = figures_dir / "table_latex.tex"
    with open(file_name_text, "w") as f:
        f.write(latex_table)


