#!/usr/bin/env python3
from itertools import product
from pathlib import Path
from path_settings import EXPERIMENTS_CONF_DIR

# Output folders
RUNS_DIR = EXPERIMENTS_CONF_DIR
MIA_DIR = EXPERIMENTS_CONF_DIR / "mia"
Flower_DIR = EXPERIMENTS_CONF_DIR / "flower"
Flower_DIR.mkdir(parents=True, exist_ok=True)
MIA_DIR.mkdir(parents=True, exist_ok=True)

# ---- Define your grids (with exact text for YAML output) ----
rounds_epochs = [
    {"num_server_rounds": "100",  "local_epochs": "1"},
    {"num_server_rounds": "20",  "local_epochs": "5"},
    {"num_server_rounds": "10", "local_epochs": "10"},
]

dists = [
    {"iid_data_distribution": "true",  "dirichlet_alpha": "0.0"},
    {"iid_data_distribution": "false", "dirichlet_alpha": "0.1"},
    {"iid_data_distribution": "false", "dirichlet_alpha": "0.5"},
]

models = [
    {"model": "simple_model",               "weight_decay": "0.0"},
    {"model": "complex_model",              "weight_decay": "0.0"},
    {"model": "simple_model_with_dropout",  "weight_decay": "5e-4"},
    {"model": "complex_model_with_dropout", "weight_decay": "5e-4"},
]

# ---- YAML render helpers (preserve order and formatting) ----
def yaml_runs(cfg: dict) -> str:
    return (
        "defaults:\n"
        "  - base\n\n"
        f"num_server_rounds: {cfg['num_server_rounds']}\n"
        f"local_epochs: {cfg['local_epochs']}\n"
        f"weight_decay: {cfg['weight_decay']}\n"
        f'model: "{cfg["model"]}"\n'
        f"iid_data_distribution: {cfg['iid_data_distribution']}\n"
        f"dirichlet_alpha: {cfg['dirichlet_alpha']}\n"
    )

def yaml_mia(run_name: str, run_idx_str: str, model_arch: str, weight_decay: str, num_server_rounds: str) -> str:
    return (
        "defaults:\n"
        "  - base\n\n"
        "parameters:\n"
        f'  model_arch: "{model_arch}"\n'
        f"  weight_decay: {weight_decay}\n"
        f'  run_name: "{run_name}"\n'
        f'  target_model_folder: "{run_idx_str}"\n'
        f'  target_model_file: "global_model_round_{num_server_rounds}.pth"\n'.format(local_epochs=num_server_rounds)
    )

# ---- Generate all combinations ----
i = 0  # start at 0 to match your earlier run1.yaml example
for re_cfg, d_cfg, m_cfg in product(rounds_epochs, dists, models):
    cfg = {**re_cfg, **d_cfg, **m_cfg}

    run_name = f"run{i}"
    run_idx_str = str(i)  # used for target_model_folder
    flower_config_path = Flower_DIR / f"{run_name}.yaml"
    mia_path = MIA_DIR / f"{run_name}.yaml"

    # Write runs config
    flower_config_path.write_text(yaml_runs(cfg))

    # Write mia config (derive fields from the same combo)
    mia_text = yaml_mia(
        run_name=run_name,
        run_idx_str=run_idx_str,
        model_arch=cfg["model"],
        weight_decay=cfg["weight_decay"],
        num_server_rounds=cfg["num_server_rounds"],
    )
    mia_path.write_text(mia_text)

    i += 1

print(f"Created {i-1} run files in {RUNS_DIR}/ and {MIA_DIR}/")