from pathlib import Path

# Root directory for the analysis
ANALYSIS_ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ANALYSIS_ROOT_DIR / "data"
FIGURES_DIR = ANALYSIS_ROOT_DIR / "figures"

# shared
WANDB_ENTITY = "kizaru-university-leipzig"
FILE_NAME_FLOWER = "flower.csv"
FILE_NAME_MIA = "mia.csv"
FILE_NAME_MERGED = "merged_flower_mia.csv"

# /// Main Experiments Settings ///
WANDB_PROJECT_FLOWER_MAIN = "flower_complete_main_experiments_run"
WANDB_PROJECT_MIA_MAIN = "mia_complete_main_experiments_run"

DATA_SUBDIR_MAIN = DATA_DIR / "main_experiments"
FIGURES_DIR_MAIN = FIGURES_DIR / "main_experiments"

# data file paths
FLOWER_DATA_PATH = DATA_SUBDIR_MAIN / FILE_NAME_FLOWER
MIA_DATA_PATH = DATA_SUBDIR_MAIN / FILE_NAME_MIA
MERGED_DATA_FILE_PATH = DATA_SUBDIR_MAIN / FILE_NAME_MERGED

# /// Side Experiments Settings ///

# --- shadow model experiments ---
WANDB_PROJECT_MIA_SIDE_SHADOW_MODELS = "mia_side_experiments_shadow_models"
DATA_SUBDIR_SHADOW = DATA_DIR / "side_experiments_shadow_models"
FIGURES_DIR_SHADOW = FIGURES_DIR / "side_experiments_shadow_models"
MIA_SIDE_SHADOW_DATA_PATH = DATA_SUBDIR_SHADOW / FILE_NAME_MIA

# --- clients experiments ---
WANDB_PROJECT_FLOWER_SIDE_CLIENTS = "flower_complete_side_experiments_run4"
WANDB_PROJECT_MIA_SIDE_CLIENTS = "mia_complete_side_experiments_run4"
DATA_SUBDIR_CLIENTS = DATA_DIR / "side_experiments_clients"
FIGURES_DIR_CLIENTS = FIGURES_DIR / "side_experiments_clients"

FLOWER_SIDE_CLIENTS_DATA_PATH = DATA_SUBDIR_CLIENTS / FILE_NAME_FLOWER
MIA_SIDE_CLIENTS_DATA_PATH = DATA_SUBDIR_CLIENTS / FILE_NAME_MIA
MERGED_SIDE_CLIENTS_DATA_FILE_PATH = DATA_SUBDIR_CLIENTS / FILE_NAME_MERGED
