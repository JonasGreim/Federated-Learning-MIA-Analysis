import os
from sklearn.utils import check_random_state
import random
import torch
import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Seed all major libraries and environment settings for full reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)  # Python hash
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # CUDA deterministic
    random.seed(seed)  # Python built-in RNG
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # Torch CPU
    torch.cuda.manual_seed(seed)  # Torch current GPU
    torch.cuda.manual_seed_all(seed)  # All GPUs
    torch.backends.cudnn.deterministic = True  # Force determinism
    torch.backends.cudnn.benchmark = False  # Disable auto-tuning
    torch.use_deterministic_algorithms(True)

    try:
        _ = check_random_state(seed)
    except ImportError:
        pass

    print(f"[Seed] Set global seed to {seed}")


def seed_worker(worker_id) -> None:
    # worker_id is automatically used internally by PyTorch to generate a unique seed for each worker process
    worker_seed = torch.initial_seed() % 2 ** 32  # worker-specific seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def release_model(model, device: str) -> None:
    """
    Safely release a PyTorch model from GPU memory.

    Moves the model to CPU, deletes the object, and clears GPU cache if needed.
    """
    if model is not None:
        model.cpu()
        del model
    if device == 'cuda':
        torch.cuda.empty_cache()
