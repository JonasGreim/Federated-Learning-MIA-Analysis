from dataclasses import dataclass
from experiments_conf_types.types_config_flower import FlowerConfig
from experiments_conf_types.types_config_mia import MiaConfig


@dataclass
class Config:
    flower: FlowerConfig
    mia: MiaConfig
    exp_name: str
    output_dir: str
