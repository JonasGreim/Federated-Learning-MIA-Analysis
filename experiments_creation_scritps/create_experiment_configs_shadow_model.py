#!/usr/bin/env python3
from path_settings import EXPERIMENTS_CONF_DIR

# Output folders
RUNS_DIR = EXPERIMENTS_CONF_DIR
MIA_DIR = EXPERIMENTS_CONF_DIR / "mia" / "shadow_models_experiments"
MIA_DIR.mkdir(parents=True, exist_ok=True)

# ---- Define your grids (with exact text for YAML output) ----
num_shadow_models = [1, 3, 10, 20, 50]

def yaml_mia(run_name: str, target_model_folder: str, model_arch: str, weight_decay: str, num_server_rounds: int, num_shadow_models: int, train_size: int) -> str:
    return (
        "defaults:\n"
        "  - base\n\n"
        "parameters:\n"
        f'  model_arch: "{model_arch}"\n'
        f"  weight_decay: {weight_decay}\n"
        f'  run_name: "{run_name}"\n'
        f'  target_model_folder: "{target_model_folder}"\n'
        f'  target_model_file: "global_model_round_{num_server_rounds}.pth"\n'
        f'  num_shadow_models: {num_shadow_models}\n'
        f'  train_size: {train_size}\n'
    )

# ---- Generate all combinations ----
i = 42  # numeration start at i
for num_shadow_cfg in num_shadow_models:

    run_name = f"run{i}"
    run_idx_str = str(i)  # used for target_model_folder
    mia_path = MIA_DIR / f"{run_name}.yaml"

    # Write mia config (derive fields from the same combo)
    mia_text = yaml_mia(
        run_name=run_name,
        target_model_folder="0",
        model_arch="simple_model",
        weight_decay="0.0",
        num_server_rounds=100,
        num_shadow_models=num_shadow_cfg,
        train_size=10000,
    )
    mia_path.write_text(mia_text)

    i += 1

print(f"Created {i-1} run files in {RUNS_DIR}/ and {MIA_DIR}/")