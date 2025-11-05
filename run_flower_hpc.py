import os, shlex
from hydra.utils import get_original_cwd
from hydra.main import main as hydra_main
from omegaconf import OmegaConf
from experiments_conf_types.types_config import Config
from experiments_conf_types.types_config_flower import FlowerConfig
from path_settings import FLOWER_BUILD_FAB_PATH


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
        f"num-clients={flower_cfg.num_clients} "
        f"run-name=\"{flower_cfg.run_name}\""
    )

    os.chdir(get_original_cwd())
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1" # disables output buffering
    env.setdefault("TERM", "dumb")

    argv = [
        "flwr", "run", str(FLOWER_BUILD_FAB_PATH), "hpc-deploy",
        "--run-config", run_config,
        "--stream",
    ]
    print("Executing:", " ".join(shlex.quote(x) for x in argv), flush=True)
    # Replace current process → SLURM captures flwr stdout/stderr directly
    os.execvpe(argv[0], argv, env)



if __name__ == "__main__":
    run()
