from pathlib import Path

# Root directory for the analysis
ANALYSIS_ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ANALYSIS_ROOT_DIR / "data"
FIGURES_DIR = ANALYSIS_ROOT_DIR / "figures"

# Wandb name settings
wandb_entity = "kizaru-university-leipzig"
wandb_project_flower_name = "flower_complete_main_experiments_run"
wandb_project_mia_name = "mia_complete_main_experiments_run"

# # shadow model experiments
# mia_side_experiments_shadow_models

# # clients experiments
# flower_complete_side_experiments_run4
# mia_complete_side_experiments_run4

file_name_flower = "main_flower"
file_name_mia = "main_mia"
DATA_SUBDIR = DATA_DIR / "main_experiments"

# data file paths
flower_data_path = DATA_SUBDIR / f"{file_name_flower}.csv"
mia_data_path = DATA_SUBDIR / f"{file_name_mia}.csv"
merge_data_file_path = DATA_SUBDIR / f"merged_{file_name_flower}_{file_name_mia}.csv"

# table file paths
overview_file_path = DATA_SUBDIR / f"overview_{file_name_flower}_{file_name_mia}.csv"
table_file_latex_path = DATA_SUBDIR / f"table_latex_{file_name_flower}_{file_name_mia}.csv"

# figures file paths
figure_dir_path = FIGURES_DIR