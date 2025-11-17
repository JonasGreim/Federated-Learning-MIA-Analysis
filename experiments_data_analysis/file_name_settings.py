from pathlib import Path

# Root directory for the analysis
ANALYSIS_ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ANALYSIS_ROOT_DIR / "data"
FIGURES_DIR = ANALYSIS_ROOT_DIR / "figures"

# Wandb name settings
wandb_entity = "kizaru-university-leipzig"
wandb_project_flower_name = "flower_complete_main_experiments_run"
wandb_project_mia_name = "mia_complete_main_experiments_run"

# data file paths
mia_data_path = DATA_DIR / f"{wandb_project_mia_name}.csv"
flower_data_path = DATA_DIR / f"{wandb_project_flower_name}.csv"
merge_data_file_path = DATA_DIR / f"merged_{wandb_project_flower_name}_{wandb_project_mia_name}.csv"

# table file paths
overview_file_path = DATA_DIR / f"overview_{wandb_project_flower_name}_{wandb_project_mia_name}.csv"
table_file_latex_path = DATA_DIR / f"table_latex_{wandb_project_flower_name}_{wandb_project_mia_name}.csv"

# figures file paths
figure_dir_path = FIGURES_DIR