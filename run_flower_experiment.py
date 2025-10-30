import os
from omegaconf import OmegaConf
from hydra.main import main as hydra_main
import ray
import gc
import subprocess
from experiments_conf_types.types_config import Config
from experiments_conf_types.types_config_flower import FlowerConfig


@hydra_main(config_path="experiments_conf", config_name="config", version_base=None)
def run(config: Config):
    # if no config is passed via command line, base config is used
    # python3 run_flower_experiment.py flower=run1

    # Load flower configuration and run flower with config parameters
    # f.e.: flwr run . --run-config 'num-server-rounds=1 local-epochs=1 ...'
    # it has to be injected through the command line into the .toml because hydra would not work with a real federated learning setup (with this current flower version)
    flower_cfg: FlowerConfig = config.flower

    os.makedirs(config.output_dir, exist_ok=True)
    OmegaConf.save(flower_cfg, os.path.join(config.output_dir, "flower_config.yaml"))

    run_config = (
        f"num-server-rounds={flower_cfg.num_server_rounds} "
        f"local-epochs={flower_cfg.local_epochs} "
        f"learning-rate={flower_cfg.learning_rate} "
        f"learning-rate-decay={flower_cfg.learning_rate_decay} "
        f"batch-size={flower_cfg.batch_size} "
        f"fraction-fit={flower_cfg.fraction_fit} "
        f"device=\"{flower_cfg.device}\" "
        f"weight-decay={flower_cfg.weight_decay} "
        f"model=\"{flower_cfg.model}\" "
        f"seed={flower_cfg.seed} "
        f"iid-data-distribution={'true' if flower_cfg.iid_data_distribution else 'false'} "  # flower expects string
        f"dirichlet-alpha={flower_cfg.dirichlet_alpha} "
        f"num-clients={flower_cfg.num_clients}"
        f"run-name={flower_cfg.run_name}"
    )
    print(f"flwr run . --run-config '{run_config}'")

    device_flag = "local-simulation-gpu" if flower_cfg.device == "cuda" else "local-simulation-cpu"
    subprocess.run(["flwr", "run", ".", device_flag, "--run-config", run_config])

    # clean up resources
    if ray.is_initialized():
        ray.shutdown()  # Make sure Ray is cleaned up if it was used
    gc.collect()  # Force garbage collection


if __name__ == "__main__":
    run()
