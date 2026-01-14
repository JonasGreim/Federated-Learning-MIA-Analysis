import pandas as pd
from experiments_data_analysis.data_interface.column_names_merged_table import Col
from pathlib import Path


def create_overview_table_clients_experiments(
        data_path: Path,
        output_path: Path,
        figures_dir: Path
):
    # Load CSV
    full_data_tables = pd.read_csv(data_path)

    keep_columns = [
        Col.MODEL, Col.NUM_CLIENTS, Col.OVERFITTING_GAP_LOSS, Col.METRICS_AUC
    ]

    overview_table = full_data_tables[keep_columns].copy()

    # Rename columns to German headers
    overview_table = overview_table.rename(columns={
        Col.MODEL: "Modell",
        Col.NUM_CLIENTS: "Anzahl Clients",
        Col.OVERFITTING_GAP_LOSS: "OGL",
        Col.METRICS_AUC: "MIA AUC",
    })

    # Ensure categorical sorting (keep your desired order)
    overview_table["Modell"] = pd.Categorical(
        overview_table["Modell"], categories=["Shokri-CNN", "ResNet-18"], ordered=True
    )
    overview_table["Anzahl Clients"] = pd.Categorical(
        overview_table["Anzahl Clients"], categories=[2, 5, 10], ordered=True
    )

    # Sort in meaningful grid order
    overview_table = overview_table.sort_values(["Modell", "Anzahl Clients"])

    # Save CSV overview
    file_name_overview = output_path / "overview.csv"
    overview_table.to_csv(file_name_overview, index=False)

    # For LaTeX grouping: set MultiIndex so multirow works (Modell grouped)
    latex_df = overview_table.set_index(["Modell", "Anzahl Clients"])

    latex_table = latex_df.to_latex(
        longtable=True,
        multirow=True,
        multicolumn=False,
        index_names=False,
        column_format="llSS",   # 2 index cols + 2 numeric cols (siunitx S)
        float_format="%.2f",
        caption="Übersicht der Experimente mit variierender Client-Anzahl.",
        label="tab:experiment_overview_clients",
        escape=False,
    )

    # ---------- Custom header (German) ----------
    correct_header = (
        "\\toprule\n"
        "Modell & Anzahl Clients & OGL & MIA AUC \\\\\n"
        "\\midrule\n"
        "\\endfirsthead\n"
        "\\caption[]{Übersicht der Experimente mit variierender Client-Anzahl.} \\\\\n"
        "\\toprule\n"
        "Modell & Anzahl Clients & OGL & MIA AUC \\\\\n"
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
        "\\textbf{OGL}: Overfitting Gap Loss, "
        "\\textbf{MIA AUC}: Area Under the ROC Curve of the Membership-Inference Attack."
        "}\n"
    )

    file_name_text = figures_dir / "table_latex_clients.tex"
    with open(file_name_text, "w") as f:
        f.write(latex_table)

    print("✅ Overview table (Clients) created successfully (German, longtable, multirow, siunitx).")
