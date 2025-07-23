import os
from omegaconf import OmegaConf
from hydra.main import main as hydra_main


@hydra_main(config_path="experiments_conf", config_name="config", version_base=None)
def run(config):
    # if no config is passed via command line, base config is used
    os.makedirs(config.output_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.output_dir, "full_config.yaml"))

    # Load flower configuration and run flower
    # flwr run . --run-config 'num-server-rounds=1 local-epochs=1'
    # it has to be injected through the command line into the .toml because hydra would not work with a real federated learning setup (with current version)
    flower_cfg = config.flower
    run_config = (
        f"num-server-rounds={flower_cfg.num_server_rounds} "
        f"local-epochs={flower_cfg.local_epochs} "
        f"weight_decay={flower_cfg.weight_decay} "
        f"fraction-fit={flower_cfg.fraction_fit} "
        f"device=\"{flower_cfg.device}\" "
        f"model=\"{flower_cfg.model}\" "
        f"iid_data_distribution={'true' if flower_cfg.iid_data_distribution else 'false'} "
        f"dirichlet_alpha={flower_cfg.dirichlet_alpha} "
        f"num_clients={flower_cfg.num_clients}"
    )
    print(f"flwr run . --run-config '{run_config}'")

    os.system(f"flwr run . --run-config '{run_config}'")


if __name__ == "__main__":
    run()
