#!/usr/bin/env python3
from path_settings import EXPERIMENTS_CONF_DIR

# Output folders
RUNS_DIR = EXPERIMENTS_CONF_DIR
MIA_DIR = EXPERIMENTS_CONF_DIR / "mia" / "client_number_mia_experiments"
Flower_DIR = EXPERIMENTS_CONF_DIR / "flower" / "client_number_mia_experiments"
Flower_DIR.mkdir(parents=True, exist_ok=True)
MIA_DIR.mkdir(parents=True, exist_ok=True)

flower_conf_simple = {
    "num_server_rounds": 100,
    "local_epochs": 1,
    "iid_data_distribution": "true",
    "dirichlet_alpha": 0.0,
    "weight_decay": 0.0
}

client_nums = [2, 5, 10]
models = ["simple_model", "complex_model"]

# ---- YAML render helpers (preserve order and formatting) ----
def yaml_runs(cfg: dict, run_name: str, client_num: int, model: str) -> str:
    return (
        "defaults:\n"
        "  - base\n\n"
        f"num_server_rounds: {cfg['num_server_rounds']}\n"
        f"local_epochs: {cfg['local_epochs']}\n"
        f"weight_decay: {cfg['weight_decay']}\n"
        f'model: "{model}"\n'
        f"iid_data_distribution: {cfg['iid_data_distribution']}\n"
        f"dirichlet_alpha: {cfg['dirichlet_alpha']}\n"
        f'run_name: "{run_name}"\n'
        f'num_clients: "{client_num}"\n'
    )

def yaml_mia(run_name: str, target_model_folder: str, model_arch: str, weight_decay: float, num_server_rounds: str) -> str:
    return (
        "defaults:\n"
        "  - base\n\n"
        "parameters:\n"
        f'  model_arch: "{model_arch}"\n'
        f"  weight_decay: {weight_decay}\n"
        f'  run_name: "{run_name}"\n'
        f'  target_model_folder: "{target_model_folder}"\n'
        f'  target_model_file: "global_model_round_{num_server_rounds}.pth"\n'
    )

# ---- Generate all combinations ----
run_id = 36  # start at 0 to match your earlier run1.yaml example
folder_name = 0
for model in models:
    for client_num in client_nums:

        run_name = f"run{run_id}"
        target_model_folder = str(folder_name)  # used for target_model_folder
        flower_config_path = Flower_DIR / f"{run_name}.yaml"
        mia_path = MIA_DIR / f"{run_name}.yaml"

        # Write runs config
        flower_config_path.write_text(yaml_runs(flower_conf_simple, run_name, client_num, model))

        # Write mia config (derive fields from the same combo)
        mia_text = yaml_mia(
            run_name=run_name,
            target_model_folder=target_model_folder,
            model_arch=model,
            weight_decay=0.0,
            num_server_rounds=flower_conf_simple.get("num_server_rounds", 100),
        )
        mia_path.write_text(mia_text)

        run_id += 1
        folder_name += 1

print(f"Created {run_id - 1} run files in {RUNS_DIR}/ and {MIA_DIR}/")



