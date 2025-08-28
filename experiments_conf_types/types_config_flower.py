from dataclasses import dataclass


@dataclass
class FlowerConfig:
    num_server_rounds: int
    local_epochs: int
    learning_rate: float
    learning_rate_decay: float
    batch_size: int
    fraction_fit: float
    device: str
    weight_decay: float
    model: str
    seed: int
    iid_data_distribution: bool
    dirichlet_alpha: float
    num_clients: int
