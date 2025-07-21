"""my-awesome-app: A Flower / PyTorch app."""
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from my_awesome_app.utils.split_cifar10_mia import split_cifar10_for_target_and_shadow
from my_awesome_app.strategies.new_strategy import FedCustom
from my_awesome_app.utils.task import get_weights, create_model, seed_everything


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config.get("fraction_fit", 1.0)  # weird error that fraction_fit is not in run_config
    learning_rate = context.run_config["learning-rate"]
    lr_decay = context.run_config.get("learning-rate-decay", 0.0)
    weight_decay = context.run_config.get("weight-decay", 0.0)
    train_target_model_as_shadow_model = context.run_config["train_target_model_as_shadow_model"]
    device = context.run_config["device"]
    seed = context.run_config.get("seed", 42)
    model_name = context.run_config.get("model")
    batch_size = context.run_config.get("batch-size", 32)

    # Seed everything for reproducibility
    seed_everything(seed)

    # Config: print the run_config dict
    print("\n[Config] run_config:")
    for k, v in context.run_config.items():
        print(f"  {k}: {v}")

    # Create dataset splits if they do not exist
    split_cifar10_for_target_and_shadow(
        target_train_ratio=0.4,
        shadow_train_ratio=0.3,
        shadow_test_ratio=0.3
    )

    # Initialize model parameters
    ndarrays = get_weights(
        create_model(model_name))  # could load check points model here (global_model_round_1) to resume training on last global model
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
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
