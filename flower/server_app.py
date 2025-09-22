from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flower.utils.data_loading import ensure_split_data_exists
from flower.utils.model_utils import get_weights, create_save_folder
from flower.utils.reproducibility import seed_everything
from flower.strategies.custom_weighted_fedavg import FedCustom
from flower.utils.wandb_logging import run_data_partitioning_for_visualization, initialize_wandb_run
from path_settings import METRICS_DIR_FLOWER
from flower.utils.model_factory import create_model


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config.get("fraction-fit", 1.0)  # weird error that fraction_fit is not in run_config
    learning_rate = context.run_config["learning-rate"]
    lr_decay = context.run_config.get("learning-rate-decay", 0.0)
    weight_decay = context.run_config.get("weight-decay", 0.0)
    train_target_model_as_shadow_model = context.run_config["train-target-model-as-shadow-model"]
    device = context.run_config["device"]
    seed = context.run_config.get("seed", 42)
    model_name = context.run_config.get("model")
    batch_size = context.run_config.get("batch-size", 32)
    # Only used for visualization and naming
    dirichlet_alpha = context.run_config.get("dirichlet-alpha", 0.5)
    iid = context.run_config.get("iid-data-distribution", True)
    num_clients = context.run_config.get("num-clients", 4)
    metric_save_folder = create_save_folder(save_dir=METRICS_DIR_FLOWER)
    print(f"Created metrics folder: {metric_save_folder}")

    # Seed everything for reproducibility
    seed_everything(seed)

    # Config: print the run_config dict
    print("\n[Config] run_config:")
    for k, v in context.run_config.items():
        print(f"  {k}: {v}")

    initialize_wandb_run(project_name="flower_target_model", run_name=f"{model_name}-{num_rounds}-{dirichlet_alpha}", config=context.run_config)

    # Create dataset splits if they do not exist
    ensure_split_data_exists()

    # Run data partitioning for visualization
    run_data_partitioning_for_visualization(iid=iid,
                                            dirichlet_alpha=dirichlet_alpha,
                                            num_partitions=num_clients,
                                            train_target_model_as_shadow_model=train_target_model_as_shadow_model,
                                            metric_save_folder=metric_save_folder,
                                            seed=seed)

    # Initialize model parameters
    ndarrays = get_weights(
        create_model(
            model_name))  # could load check points model here (global_model_round_1) to resume training on last global model
    parameters = ndarrays_to_parameters(ndarrays)

    strategy = FedCustom(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        learning_rate=learning_rate,
        lr_decay=lr_decay,
        weight_decay=weight_decay,
        train_target_model_as_shadow_model=train_target_model_as_shadow_model,
        device=device,
        model_name=model_name,
        batch_size=batch_size,
        max_server_rounds=num_rounds,
        metric_save_folder=metric_save_folder,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
