"""my-awesome-app: A Flower / PyTorch app."""
from typing import List, Tuple
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from my_awesome_app.split_cifar10_mia import create_split_files
from my_awesome_app.strategies.new_strategy import FedCustom
from my_awesome_app.task import get_weights, create_model


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["learning-rate"]
    learning_rate = context.run_config["learning-rate"]
    split_flag = context.run_config["dataset-split"]

    # Create dataset splits if they do not exist
    create_split_files()

    # Initialize model parameters
    ndarrays = get_weights(
        create_model())  # could load check points model here (global_model_round_1) to resume training on last global model
    parameters = ndarrays_to_parameters(ndarrays)

    strategy2 = FedCustom(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        learning_rate=learning_rate,
        dataset_split=split_flag,
        device='cpu',
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy2, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
