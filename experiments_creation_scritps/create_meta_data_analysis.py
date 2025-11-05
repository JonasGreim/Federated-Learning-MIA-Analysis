#!/usr/bin/env python3
import json
from itertools import product
from path_settings import EXPERIMENTS_CONF_DIR, EXPERIMENTS_ANALYSIS_DIR_DATA

# folders
RUNS_DIR = EXPERIMENTS_CONF_DIR
Flower_DIR = EXPERIMENTS_CONF_DIR / "flower" / "main_mia_experiments"
DATA_ANALYSIS_DIR = EXPERIMENTS_ANALYSIS_DIR_DATA / "big_run"

Flower_DIR.mkdir(parents=True, exist_ok=True)
DATA_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# ---- Define your grids ----
rounds_epochs = [
    {"num_server_rounds": 100, "local_epochs": 1},
    {"num_server_rounds": 20, "local_epochs": 5},
    {"num_server_rounds": 10, "local_epochs": 10},
]

dists = [
    {"iid_data_distribution": True, "dirichlet_alpha": 0.0},
    {"iid_data_distribution": False, "dirichlet_alpha": 0.1},
    {"iid_data_distribution": False, "dirichlet_alpha": 0.5},
]

models = [
    {"architecture": "shokri_cnn", "with_protection": False},
    {"architecture": "resnet_18", "with_protection": False},
    {"architecture": "shokri_cnn", "with_protection": True},
    {"architecture": "resnet_18", "with_protection": True},
]

# ---- JSON render helper ----
def json_run(cfg: dict, run_name: str) -> str:
    cfg_with_name = {**cfg, "run_name": run_name}
    return json.dumps(cfg_with_name, indent=2)

# ---- Generate all combinations ----
i = 0  # start at 0 to match your earlier run1.yaml example
for re_cfg, d_cfg, m_cfg in product(rounds_epochs, dists, models):
    cfg = {**re_cfg, **d_cfg, **m_cfg}

    run_name = f"run{i}"
    experiment_data_path = DATA_ANALYSIS_DIR / f"{run_name}" / "metadata.json"

    # Write JSON run config
    experiment_data_path.write_text(json_run(cfg, run_name))

    i += 1

print(f"Created {i} JSON run files in {DATA_ANALYSIS_DIR}")
