from hydra.main import main as hydra_main
import os
from omegaconf import OmegaConf
from membership_inference_attack.mia_shokri import run_mia
from experiments_conf_types.types_config_mia import MiaConfig
from experiments_conf_types.types_config import Config


@hydra_main(config_path="experiments_conf", config_name="config", version_base=None)
def run_mia_experiment(config: Config):
    # if no config is passed via command line, base config is used
    # python3 run_mia_experiment.py mia=run1
    mia_config: MiaConfig = config.mia

    os.makedirs(config.output_dir, exist_ok=True)
    OmegaConf.save(mia_config, os.path.join(config.output_dir, "mia_config.yaml"))

    print(OmegaConf.to_yaml(config.mia))

    run_mia(mia_config)


if __name__ == "__main__":
    run_mia_experiment()
