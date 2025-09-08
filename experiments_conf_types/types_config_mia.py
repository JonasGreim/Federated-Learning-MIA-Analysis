from dataclasses import dataclass


@dataclass
class Parameters:
    num_classes: int
    num_shadow_models: int
    shadow_epochs: int
    model_arch: str
    train_size: int
    test_size: int
    weight_decay: float
    run_name: str
    target_model_folder: str
    target_model_file: str


@dataclass
class ParametersStatic:
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
