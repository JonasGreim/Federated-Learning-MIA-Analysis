import os

# Root of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to various directories
MY_APP_DIR = os.path.join(ROOT_DIR, "my_awesome_app")
IMAGES_DIR = os.path.join(ROOT_DIR, "images")
METRICS_DIR = os.path.join(ROOT_DIR, "metrics_of_run")
CHECKPOINTS_DIR_Target = os.path.join(ROOT_DIR, "model_checkpoints_target")
CHECKPOINTS_DIR_Shadow = os.path.join(ROOT_DIR, "model_checkpoints_shadow")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
WANDB_DIR = os.path.join(ROOT_DIR, "wandb")
SPLITS_DIR = os.path.join(ROOT_DIR, "splits")

# Paths to config or metadata files
PYPROJECT_PATH = os.path.join(ROOT_DIR, "pyproject.toml")

# Example: Path to a specific split
D1_SPLIT_PATH = os.path.join(SPLITS_DIR, "D1")
D2_SPLIT_PATH = os.path.join(SPLITS_DIR, "D2")
D3_SPLIT_PATH = os.path.join(SPLITS_DIR, "D3")
D4_SPLIT_PATH = os.path.join(SPLITS_DIR, "D4")


# Utility function (optional): ensure a directory exists
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
