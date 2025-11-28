import os
from experiments_data_analysis.create_dataset.merge_flower_mia_data import merge_flower_mia_wandb_runs_to_csv
from experiments_data_analysis.create_dataset.wandb_get_flower_data import export_flower_wandb_runs_to_csv
from experiments_data_analysis.create_dataset.wandb_get_mia_data import export_mia_wandb_runs_to_csv
from experiments_data_analysis.create_figures.side_experiments_clients.create_client_num_figures import create_client_num_figures
from experiments_data_analysis.create_figures.side_experiments_clients.create_overview_table import \
    create_overview_table_clients_experiments
from experiments_data_analysis.create_figures.side_experiments_shadow_models.create_overview_table import \
    create_overview_table_shadow_experiments
from experiments_data_analysis.create_figures.side_experiments_shadow_models.create_shadow_model_number_figures import \
    create_shadow_model_number_figures
from experiments_data_analysis.file_name_settings import WANDB_ENTITY, FIGURES_DIR, \
    WANDB_PROJECT_MIA_SIDE_SHADOW_MODELS, DATA_SUBDIR_SHADOW, \
    WANDB_PROJECT_MIA_SIDE_CLIENTS, DATA_SUBDIR_CLIENTS, WANDB_PROJECT_FLOWER_SIDE_CLIENTS, MIA_SIDE_CLIENTS_DATA_PATH, \
    FLOWER_SIDE_CLIENTS_DATA_PATH, MERGED_SIDE_CLIENTS_DATA_FILE_PATH, FIGURES_DIR_CLIENTS, MIA_SIDE_SHADOW_DATA_PATH, \
    FIGURES_DIR_SHADOW, MIA_SIDE_SHADOW_DATA_OVERVIEW
from plot_style import use_thesis_style

use_thesis_style()


# Ensure figures directory exists
os.makedirs(DATA_SUBDIR_SHADOW, exist_ok=True)
os.makedirs(DATA_SUBDIR_CLIENTS, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR_CLIENTS, exist_ok=True)
os.makedirs(FIGURES_DIR_SHADOW, exist_ok=True)

# --- shadow model experiments ---
# Export flower wandb runs to CSV
export_mia_wandb_runs_to_csv(entity=WANDB_ENTITY, project=WANDB_PROJECT_MIA_SIDE_SHADOW_MODELS,
                             output_path=DATA_SUBDIR_SHADOW)
create_overview_table_shadow_experiments(
        data_path=MIA_SIDE_SHADOW_DATA_PATH,
        output_path=DATA_SUBDIR_SHADOW,
        figures_dir=FIGURES_DIR_SHADOW
)

create_shadow_model_number_figures(data_path=MIA_SIDE_SHADOW_DATA_OVERVIEW, figures_dir=FIGURES_DIR_SHADOW)

# --- clients experiments ---
# Export flower wandb runs to CSV
export_flower_wandb_runs_to_csv(entity=WANDB_ENTITY, project=WANDB_PROJECT_FLOWER_SIDE_CLIENTS,
                                output_path=DATA_SUBDIR_CLIENTS)
export_mia_wandb_runs_to_csv(entity=WANDB_ENTITY, project=WANDB_PROJECT_MIA_SIDE_CLIENTS,
                             output_path=DATA_SUBDIR_CLIENTS)
# Merge both CSVs
merge_flower_mia_wandb_runs_to_csv(
    mia_data_path=MIA_SIDE_CLIENTS_DATA_PATH,
    flower_data_path=FLOWER_SIDE_CLIENTS_DATA_PATH,
    output_path=DATA_SUBDIR_CLIENTS,
)

create_client_num_figures(data_path=MERGED_SIDE_CLIENTS_DATA_FILE_PATH, figures_dir=FIGURES_DIR_CLIENTS)
create_overview_table_clients_experiments(data_path=MERGED_SIDE_CLIENTS_DATA_FILE_PATH, output_path=DATA_SUBDIR_CLIENTS, figures_dir=FIGURES_DIR_CLIENTS)
