from experiments_data_analysis.create_dataset.merge_flower_mia_data import merge_flower_mia_wandb_runs_to_csv
from experiments_data_analysis.create_dataset.wandb_get_flower_data import export_flower_wandb_runs_to_csv
from experiments_data_analysis.create_dataset.wandb_get_mia_data import export_mia_wandb_runs_to_csv
from experiments_data_analysis.create_figures.create_overview_tables import create_overview_table
from experiments_data_analysis.create_figures.seaborn_figures import create_analysis_figures
from experiments_data_analysis.create_figures.seaborn_figures_corr import create_analysis_figures_corr
from experiments_data_analysis.file_name_settings import wandb_project_flower_name, mia_data_path, flower_data_path, \
    merge_data_file_path, figure_dir_path, overview_file_path, table_file_latex_path, DATA_SUBDIR
from experiments_data_analysis.file_name_settings import wandb_entity, wandb_project_mia_name
import os

# Ensure figures directory exists
os.makedirs(DATA_SUBDIR, exist_ok=True)
os.makedirs(figure_dir_path, exist_ok=True)

# Export flower wandb runs to CSV
export_flower_wandb_runs_to_csv(entity=wandb_entity, project=wandb_project_flower_name, output_path=flower_data_path)

# Export flower wandb runs to CSV
export_mia_wandb_runs_to_csv(entity=wandb_entity, project=wandb_project_mia_name, output_path=mia_data_path)

# Merge both CSVs
merge_flower_mia_wandb_runs_to_csv(
    mia_data_path=mia_data_path,
    flower_data_path=flower_data_path,
    output_path=merge_data_file_path,
)

# create figures
create_analysis_figures(data_path=merge_data_file_path, figures_dir=figure_dir_path)
create_analysis_figures_corr(data_path=merge_data_file_path, figures_dir=figure_dir_path)

# create overview table
create_overview_table(data_path=merge_data_file_path, output_path=overview_file_path,
                      output_path_latex=table_file_latex_path)
