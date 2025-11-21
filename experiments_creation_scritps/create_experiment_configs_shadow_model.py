#!/usr/bin/env python3
from path_settings import EXPERIMENTS_CONF_DIR

# Output folders
RUNS_DIR = EXPERIMENTS_CONF_DIR
MIA_DIR = EXPERIMENTS_CONF_DIR / "mia" / "shadow_models_experiments"
MIA_DIR.mkdir(parents=True, exist_ok=True)

# ---- Define your grids (with exact text for YAML output) ----
num_shadow_models = [1, 3, 10, 20]
model_architectures = ["simple_model", "complex_model"]
data_distributions = ["iid", "non-iid"]


def yaml_mia(run_name: str, target_model_folder: str, model_arch: str, weight_decay: str, num_server_rounds: int,
             num_shadow_models: int, train_size: int) -> str:
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
run_id = 42  # numeration start at i
target_model_path = [0, 4, 1, 5]  # use model checkpoints from main runs
i = 0
for model_architecture in model_architectures:
    for data_distribution in data_distributions:
        for num_shadow_cfg in num_shadow_models:

            run_name = f"run{run_id}"
            run_idx_str = str(run_id)  # used for target_model_folder
            mia_path = MIA_DIR / f"{run_name}.yaml"

            # Write mia config (derive fields from the same combo)
            if num_shadow_cfg == 1:
                train_size = 15000
            else:
                train_size = 10000

            mia_text = yaml_mia(
                run_name=run_name,
                target_model_folder=f"{target_model_path[i]}",
                model_arch=model_architecture,
                weight_decay="0.0",
                num_server_rounds=100,
                num_shadow_models=num_shadow_cfg,
                train_size=train_size,
            )
            mia_path.write_text(mia_text)

            run_id += 1
        i += 1

print(f"Created {i} run files in {RUNS_DIR}/ and {MIA_DIR}/")
