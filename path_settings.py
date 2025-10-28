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
# CHECKPOINTS_DIR_TARGET = ROOT_DIR / "model_checkpoints_target"
CHECKPOINTS_DIR_TARGET = Path("/work/hi85udaj-flower") / "model_checkpoints_target_complex"
OUTPUTS_DIR = ROOT_DIR / "outputs"
WANDB_DIR = ROOT_DIR / "wandb"

# local simulation dataset splits directory
# SPLITS_DIR = ROOT_DIR / "dataset_splits"
SHARED_NFS_PATH = "/work/hi85udaj-flower/dataset_splits"
SPLITS_DIR_STR = os.getenv("SPLITS_DIR", SHARED_NFS_PATH)
SPLITS_DIR = Path(SPLITS_DIR_STR)


EXPERIMENTS_CONF_DIR = ROOT_DIR / "experiments_conf"

# Config or metadata paths
PYPROJECT_PATH = ROOT_DIR / "pyproject.toml"

# Specific splits
D1_SPLIT_PATH = SPLITS_DIR / "D1"
D2_SPLIT_PATH = SPLITS_DIR / "D2"
D3_SPLIT_PATH = SPLITS_DIR / "D3"
D4_SPLIT_PATH = SPLITS_DIR / "D4"


# Utility
def ensure_dir_exist(path: Path):
    path.mkdir(parents=True, exist_ok=True)
