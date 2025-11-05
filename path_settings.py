from pathlib import Path
import os

# Root directory of the project
ROOT_DIR = Path(__file__).resolve().parent

# Subdirectories
MY_APP_DIR = ROOT_DIR / "flower"
IMAGES_DIR = ROOT_DIR / "images"
METRICS_DIR = ROOT_DIR / "metrics"
METRICS_DIR_FLOWER = METRICS_DIR / "flower"
METRICS_DIR_MIA = METRICS_DIR / "mia"
OUTPUTS_DIR = ROOT_DIR / "outputs"
WANDB_DIR = ROOT_DIR / "wandb"
EXPERIMENTS_CONF_DIR = ROOT_DIR / "experiments_conf"
PYPROJECT_PATH = ROOT_DIR / "pyproject.toml"

# HPC Configs
SHARED_NFS_PATH = "/work/hi85udaj-flower/dataset_splits"
SPLITS_DIR_STR = os.getenv("SPLITS_DIR", SHARED_NFS_PATH)
SPLITS_DIR = Path(SPLITS_DIR_STR)
CHECKPOINTS_DIR_TARGET = Path("/work/hi85udaj-flower") / "all_models_complete_run"
FLOWER_BUILD_DIR = Path("/work/hi85udaj-flower/build")

# Local Simulation Configs
# SPLITS_DIR = ROOT_DIR / "dataset_splits"
# CHECKPOINTS_DIR_TARGET = ROOT_DIR / "model_checkpoints_target"
# FLOWER_BUILD_DIR = ROOT_DIR / "build"


# Specific splits
D1_SPLIT_PATH = SPLITS_DIR / "D1"
D2_SPLIT_PATH = SPLITS_DIR / "D2"
D3_SPLIT_PATH = SPLITS_DIR / "D3"
D4_SPLIT_PATH = SPLITS_DIR / "D4"

# build path
FLOWER_BUILD_FAB_PATH = FLOWER_BUILD_DIR / "flower_app.fab"

# Utility
def ensure_dir_exist(path: Path):
    path.mkdir(parents=True, exist_ok=True)
