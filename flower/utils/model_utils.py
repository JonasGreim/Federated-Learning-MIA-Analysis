import torch.nn as nn
import numpy as np
from collections import OrderedDict
from typing import List
import torch
import os
from pathlib import Path


def get_weights(net: nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net: nn.Module, parameters: List[np.ndarray]) -> None:
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def create_model_save_folder(save_dir: Path) -> Path:
    """Creates each run a unique folder in the save_dir for saving model checkpoints (0, 1, ...)"""
    counter = 0
    folder_name = "0"
    while os.path.exists(os.path.join(save_dir, folder_name)):
        counter += 1
        folder_name = str(counter)
    model_folder_path = os.path.join(save_dir, folder_name)
    os.makedirs(model_folder_path)
    print(f"Created model folder: {model_folder_path}")

    return Path(model_folder_path)
