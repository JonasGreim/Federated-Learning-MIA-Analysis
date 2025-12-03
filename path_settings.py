from pathlib import Path

# Root directory of the project
ROOT_DIR = Path(__file__).resolve().parent

# wandb project name
WANDB_PROJECT_NAME_FLOWER = "flower_complete_side_experiments_clients_re_run6"
WANDB_PROJECT_NAME_MIA = "mia_complete_side_experiments_clients_re_run6"

# HPC Configs
PERSISTENT_MEMORY_DIR = Path("/work/hi85udaj-flower")
SPLITS_DIR = PERSISTENT_MEMORY_DIR / "dataset_splits"
CHECKPOINTS_DIR_TARGET = PERSISTENT_MEMORY_DIR / "all_models_full_side_experiments6"
METRICS_DIR = PERSISTENT_MEMORY_DIR / "metrics5Clients_side_experiments_clients6"

# Local Simulation Configs
# SPLITS_DIR = ROOT_DIR / "dataset_splits"
# CHECKPOINTS_DIR_TARGET = ROOT_DIR / "model_checkpoints_target"
# METRICS_DIR = ROOT_DIR / "metrics"

# Subdirectories
MY_APP_DIR = ROOT_DIR / "flower"
OUTPUTS_DIR = ROOT_DIR / "outputs"
WANDB_DIR = ROOT_DIR / "wandb"
EXPERIMENTS_CONF_DIR = ROOT_DIR / "experiments_conf"
PYPROJECT_PATH = ROOT_DIR / "pyproject.toml"
EXPERIMENTS_ANALYSIS_DIR_DATA = ROOT_DIR / "experiments_data_analysis" / "run_data"
METRICS_DIR_FLOWER = METRICS_DIR / "flower"
METRICS_DIR_MIA = METRICS_DIR / "mia"


# Specific splits
# D1=target train set, D2=target test set, D3=shadow train set, D4=shadow test set
D1_SPLIT_PATH = SPLITS_DIR / "D1"
D2_SPLIT_PATH = SPLITS_DIR / "D2"
D3_SPLIT_PATH = SPLITS_DIR / "D3"
D4_SPLIT_PATH = SPLITS_DIR / "D4"


# Utility
def ensure_dir_exist(path: Path):
    path.mkdir(parents=True, exist_ok=True)
