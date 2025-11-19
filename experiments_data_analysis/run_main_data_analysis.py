from experiments_data_analysis.create_dataset.merge_flower_mia_data import merge_flower_mia_wandb_runs_to_csv
from experiments_data_analysis.create_dataset.wandb_get_flower_data import export_flower_wandb_runs_to_csv
from experiments_data_analysis.create_dataset.wandb_get_mia_data import export_mia_wandb_runs_to_csv
from experiments_data_analysis.create_figures.main_experiments.create_overview_tables import create_overview_table
from experiments_data_analysis.create_figures.main_experiments.seaborn_figures import create_analysis_figures
from experiments_data_analysis.create_figures.main_experiments.seaborn_figures_corr import create_analysis_figures_corr
from experiments_data_analysis.file_name_settings import WANDB_PROJECT_FLOWER_MAIN, DATA_SUBDIR_MAIN, FIGURES_DIR, \
    FLOWER_DATA_PATH, MIA_DATA_PATH, MERGED_DATA_FILE_PATH, FIGURES_DIR_MAIN
from experiments_data_analysis.file_name_settings import WANDB_ENTITY, WANDB_PROJECT_MIA_MAIN
import os

# Ensure figures directory exists
os.makedirs(DATA_SUBDIR_MAIN, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR_MAIN, exist_ok=True)

# Export flower wandb runs to CSV
export_flower_wandb_runs_to_csv(entity=WANDB_ENTITY, project=WANDB_PROJECT_FLOWER_MAIN, output_path=DATA_SUBDIR_MAIN)

# Export mia wandb runs to CSV
export_mia_wandb_runs_to_csv(entity=WANDB_ENTITY, project=WANDB_PROJECT_MIA_MAIN, output_path=DATA_SUBDIR_MAIN)

# Merge both CSVs
merge_flower_mia_wandb_runs_to_csv(
    mia_data_path=MIA_DATA_PATH,
    flower_data_path=FLOWER_DATA_PATH,
    output_path=DATA_SUBDIR_MAIN,
)

# create figures
create_analysis_figures(data_path=MERGED_DATA_FILE_PATH, figures_dir=FIGURES_DIR_MAIN)
create_analysis_figures_corr(data_path=MERGED_DATA_FILE_PATH, figures_dir=FIGURES_DIR_MAIN)

# create overview table
create_overview_table(data_path=MERGED_DATA_FILE_PATH, output_path=DATA_SUBDIR_MAIN, figures_dir=FIGURES_DIR_MAIN)
