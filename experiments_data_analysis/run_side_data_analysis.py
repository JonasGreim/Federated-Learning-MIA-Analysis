import os
from experiments_data_analysis.create_dataset.wandb_get_flower_data import export_flower_wandb_runs_to_csv
from experiments_data_analysis.create_dataset.wandb_get_mia_data import export_mia_wandb_runs_to_csv
from experiments_data_analysis.file_name_settings import WANDB_ENTITY, FIGURES_DIR, \
    WANDB_PROJECT_MIA_SIDE_SHADOW_MODELS, DATA_SUBDIR_SHADOW, \
    WANDB_PROJECT_MIA_SIDE_CLIENTS, DATA_SUBDIR_CLIENTS, WANDB_PROJECT_FLOWER_SIDE_CLIENTS

# Ensure figures directory exists
os.makedirs(DATA_SUBDIR_SHADOW, exist_ok=True)
os.makedirs(DATA_SUBDIR_CLIENTS, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# --- shadow model experiments ---
# Export flower wandb runs to CSV
export_mia_wandb_runs_to_csv(entity=WANDB_ENTITY, project=WANDB_PROJECT_MIA_SIDE_SHADOW_MODELS,
                             output_path=DATA_SUBDIR_SHADOW)

# --- clients experiments ---
# Export flower wandb runs to CSV
export_flower_wandb_runs_to_csv(entity=WANDB_ENTITY, project=WANDB_PROJECT_FLOWER_SIDE_CLIENTS,
                             output_path=DATA_SUBDIR_CLIENTS)
export_mia_wandb_runs_to_csv(entity=WANDB_ENTITY, project=WANDB_PROJECT_MIA_SIDE_CLIENTS,
                             output_path=DATA_SUBDIR_CLIENTS)
