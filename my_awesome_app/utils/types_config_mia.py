from dataclasses import dataclass


@dataclass
class Paths:
    """Configuration for file paths used in the MIA."""
    target_checkpoint_dir: str
    data_dir: str
    split_dir: str
    current_root: str


@dataclass
class Parameters:
    """Configuration for MIA parameters."""
    num_classes: int

    num_shadow_models: int
    shadow_epochs: int
    model_arch: str

    train_size: int
    test_size: int


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
    paths: Paths
    parameters: Parameters
    parameters_static: ParametersStatic
