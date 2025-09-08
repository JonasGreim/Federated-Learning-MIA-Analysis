from flower.utils.reproducibility import release_model
from path_settings import CHECKPOINTS_DIR_TARGET
import torch
import torch.nn as nn
from flower.utils.model_factory import create_model

def load_latest_target_model(model_name: str, device: torch.device) -> nn.Module:
    release_model(None, device.type)
    model = create_model(model_name)

    # Get all checkpoint paths
    checkpoint_dir_target = CHECKPOINTS_DIR_TARGET

    numeric_dirs = [
        int(p.name) for p in checkpoint_dir_target.iterdir()
        if p.is_dir() and p.name.isdigit()
    ]
    if numeric_dirs:
        highest_folder_number = max(numeric_dirs)
    else:
        raise FileNotFoundError("No checkpoint folder found.")

    checkpoint_dir = checkpoint_dir_target / str(highest_folder_number)
    checkpoint_paths = list(checkpoint_dir.glob("global_model_round_*.pth"))

    if not checkpoint_paths:
        raise FileNotFoundError("No checkpoint files found.")

    # Extract round numbers and map to paths
    checkpoints_with_rounds = []
    for path in checkpoint_paths:
        match = re.search(r'global_model_round_(\d+)\.pth', path.name)
        if match:
            round_number = int(match.group(1))
            checkpoints_with_rounds.append((round_number, path))

    if not checkpoints_with_rounds:
        raise ValueError("No valid checkpoint files with round numbers found.")

    # Get the path with the highest round number
    latest_round, latest_checkpoint_path = max(checkpoints_with_rounds, key=lambda x: x[0])

    model.load_state_dict(torch.load(latest_checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded target model from: {latest_checkpoint_path} (round {latest_round})")

    return model