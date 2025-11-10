import torch.nn as nn
import numpy as np
from collections import OrderedDict
from typing import List
import torch
import os
from pathlib import Path
from path_settings import ensure_dir_exist
from datetime import datetime


def get_weights(net: nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net: nn.Module, parameters: List[np.ndarray]) -> None:
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def create_model_checkpoint_save_folder(save_dir: Path) -> Path:
    """Creates each run a unique folder in the model_checkpoint_target for saving .pth"""
    ensure_dir_exist(save_dir)
    counter = 0
    folder_name = "0"
    while os.path.exists(os.path.join(save_dir, folder_name)):
        counter += 1
        folder_name = str(counter)
    folder_path = os.path.join(save_dir, folder_name)
    os.makedirs(folder_path)

    return Path(folder_path)

def create_metric_save_folder(save_dir: Path, run_name: str) -> Path:
    """Creates a unique timestamped folder in metrics for each run."""
    os.makedirs(save_dir, exist_ok=True)

    # Create timestamp string, e.g., 2025-11-10_14-32
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # Combine run name with timestamp
    folder_name = f"{run_name}__{timestamp}"
    folder_path = Path(save_dir) / folder_name

    # Create the folder
    os.makedirs(folder_path, exist_ok=True)

    return folder_path

