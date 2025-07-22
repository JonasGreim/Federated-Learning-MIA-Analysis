import os
from omegaconf import OmegaConf
from hydra.main import main as hydra_main
from hydra import initialize, compose

from membership_inference_attack.mia_shokri import run_mia
from membership_inference_attack.utils.types_config_mia import MiaConfig


@hydra_main(config_path="experiments_conf", config_name="config", version_base=None)
def run(cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.output_dir, "full_config.yaml"))

    # Load flower configuration and run flower
    flower_cfg = cfg.flower
    run_config = (
        f"\"num-server-rounds\" = {flower_cfg.num_server_rounds} "
        f"\"local-epochs\" = {flower_cfg.local_epochs} "
        f"\"weight_decay\" = {flower_cfg.weight_decay} "
        f"\"fraction-fit\" = {flower_cfg.fraction_fit} "
        f"\"device\" = \"{flower_cfg.device}\" "
        f"\"model\" = \"{flower_cfg.model}\" "
        f"\"iid_data_distribution\" = {'true' if flower_cfg.iid_data_distribution else 'false'} "
        f"\"dirichlet_alpha\" = {flower_cfg.dirichlet_alpha} "
        f"\"num_clients\" = {flower_cfg.num_clients}"
    )

    os.system(f'flwr run . --run-config "{run_config}"')

    # Load mia configuration and run mia
    print("[INFO] Flower finished, now launching MIA...")
    #
    # mia_config_name = cfg.mia if isinstance(cfg.mia, str) else "base"
    # with initialize(config_path="configs_mia", version_base=None):
    #     mia_cfg_raw = compose(config_name=mia_config_name)
    # mia_cfg_dict = OmegaConf.to_container(mia_cfg_raw, resolve=True)
    # typed_mia_cfg = MiaConfig(**mia_cfg_dict)
    #
    # run_mia(typed_mia_cfg)


if __name__ == "__main__":
    run()
