import pandas as pd
from pathlib import Path

from experiments_data_analysis.data_interface.column_names_merged_table import Col
from experiments_data_analysis.create_dataset.merge_utils import map_model_name
from experiments_data_analysis.create_figures.side_experiments_shadow_models.shadow_models_utils import map_run_name


def create_overview_table_shadow_experiments(
        data_path: Path,
        output_path: Path,
        figures_dir: Path
):
    # Load CSV
    full_data_tables = pd.read_csv(data_path)

    # Rename columns to match interface
    full_data_tables = full_data_tables.rename(columns={
        "model_arch": Col.MODEL,
        "run_name": Col.RUN_NAME,
    })

    keep_columns = [
        Col.RUN_NAME,  # needed for mapping, will be dropped afterwards
        Col.MODEL,
        Col.NUM_SHADOW_MODELS,
        Col.TRAIN_SIZE,
        Col.METRICS_AUC,
    ]

    overview_table = full_data_tables[keep_columns].copy()

    # Map model names for readability
    overview_table[Col.MODEL] = overview_table[Col.MODEL].apply(map_model_name)

    # Derive data distribution from run name
    overview_table[Col.DATA_DISTRIBUTION] = overview_table[Col.RUN_NAME].apply(map_run_name)

    # Drop run name (requested)
    overview_table = overview_table.drop(columns=[Col.RUN_NAME])

    # Rename columns to short German headers
    overview_table = overview_table.rename(columns={
        Col.MODEL: "Mod",
        Col.DATA_DISTRIBUTION: "Dist",
        Col.NUM_SHADOW_MODELS: "Nsh",
        Col.TRAIN_SIZE: "Ntr",
        Col.METRICS_AUC: "AUC",
    })

    # --- Robust typing / formatting (prevents LaTeX/siunitx headaches) ---
    # Ensure integers where appropriate
    overview_table["Nsh"] = pd.to_numeric(overview_table["Nsh"], errors="coerce").astype("Int64")
    overview_table["Ntr"] = pd.to_numeric(overview_table["Ntr"], errors="coerce").astype("Int64")
    overview_table["AUC"] = pd.to_numeric(overview_table["AUC"], errors="coerce")

    # Categorical sorting (adjust categories if your map_run_name returns other labels)
    overview_table["Mod"] = pd.Categorical(
        overview_table["Mod"], categories=["Shokri-CNN", "ResNet-18"], ordered=True
    )
    overview_table["Dist"] = pd.Categorical(
        overview_table["Dist"],
        categories=["IID", "semi-non-IID", "non-IID"],
        ordered=True
    )

    # Sort: model grouped, then distribution, then number of shadow models
    overview_table = overview_table.sort_values(
        ["Mod", "Dist", "Nsh"],
        ascending=[True, True, True]
    )

    # Save CSV overview
    file_name_overview = output_path / "overview.csv"
    overview_table.to_csv(file_name_overview, index=False)

    # For LaTeX grouping (multirow on Mod)
    latex_df = overview_table.set_index(["Mod", "Dist", "Nsh"])

    # Important: multirow creates empty cells -> keep numeric columns as strings in LaTeX output
    latex_df_out = latex_df.copy()
    latex_df_out["Ntr"] = latex_df_out["Ntr"].map(lambda x: "" if pd.isna(x) else f"{int(x)}")
    latex_df_out["AUC"] = latex_df_out["AUC"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")

    latex_table = latex_df_out.to_latex(
        longtable=True,
        multirow=True,
        multicolumn=False,
        index_names=False,
        column_format="lllrr",   # 3 index cols + 2 numeric cols (NO siunitx -> no 'Invalid number' ever)
        escape=False,
        caption="Übersicht der Shadow-Model-Experimente.",
        label="tab:experiment_overview_shadow_models",
    )

    # ---------- Custom header (short) ----------
    correct_header = (
        "\\toprule\n"
        "Mod & Dist & Nsh & Ntr & AUC \\\\\n"
        "\\midrule\n"
        "\\endfirsthead\n"
        "\\caption[]{Übersicht der Shadow-Model-Experimente.} \\\\\n"
        "\\toprule\n"
        "Mod & Dist & Nsh & Ntr & AUC \\\\\n"
        "\\midrule\n"
        "\\endhead\n"
    )

    auto_head_start = "\\toprule"
    auto_head_end = "\\endhead"

    start_idx = latex_table.find(auto_head_start)
    end_idx = latex_table.find(auto_head_end) + len(auto_head_end)
    auto_header_block = latex_table[start_idx:end_idx]
    latex_table = latex_table.replace(auto_header_block, correct_header, 1)

    # ---------- Continuation text in German ----------
    latex_table = latex_table.replace(
        "Continued on next page", "Fortsetzung auf der nächsten Seite."
    ).replace(
        "continued from previous page", "Fortsetzung von der vorherigen Seite."
    )

    # ---------- Explanatory note under the table ----------
    latex_table = latex_table.replace(
        "\\end{longtable}",
        "\\end{longtable}\n"
        "\\footnotesize{"
        "\\textbf{AUC}: Area Under the ROC Curve (Membership-Inference Attack). "
        "\\textbf{Ntr}: Trainingsgröße pro Shadow Model, "
        "\\textbf{Nsh}: Anzahl Shadow-Modelle."
        "}\n"
    )

    # Write LaTeX table
    file_name_text = figures_dir / "table_latex_shadow_models.tex"
    with open(file_name_text, "w") as f:
        f.write(latex_table)

    print("✅ Overview table created (German, longtable, multirow, short headers, robust).")
