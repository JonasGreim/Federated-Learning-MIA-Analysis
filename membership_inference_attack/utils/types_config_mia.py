from dataclasses import dataclass


@dataclass
class Parameters:
    """Configuration for MIA parameters."""
    num_classes: int
    num_shadow_models: int
    shadow_epochs: int
    model_arch: str
    train_size: int
    test_size: int
    weight_decay: float
    run_name: str
    iid_data_distribution: bool
    dirichlet_alpha: float

@dataclass
class ParametersStatic:
    """Configuration for MIA parameters."""
    seed: int
    batch_size: int
    device: str
    learning_rate_decay: float
    learning_rate: float
    num_workers: int
    class_names: list[str]


@dataclass
class MiaConfig:
    parameters: Parameters
    parameters_static: ParametersStatic
