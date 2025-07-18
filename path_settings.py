from pathlib import Path

# Root directory of the project
ROOT_DIR = Path(__file__).resolve().parent

# Subdirectories
MY_APP_DIR = ROOT_DIR / "my_awesome_app"
IMAGES_DIR = ROOT_DIR / "images"
METRICS_DIR = ROOT_DIR / "metrics_of_run"
CHECKPOINTS_DIR_TARGET = ROOT_DIR / "model_checkpoints_target"
CHECKPOINTS_DIR_SHADOW = ROOT_DIR / "model_checkpoints_shadow"
OUTPUTS_DIR = ROOT_DIR / "outputs"
WANDB_DIR = ROOT_DIR / "wandb"
SPLITS_DIR = ROOT_DIR / "splits"

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
