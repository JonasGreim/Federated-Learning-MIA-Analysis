"""my-awesome-app: A Flower / PyTorch app."""
import math
from typing import List, Tuple
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from my_awesome_app.task import Net, get_weights, set_weights, test, get_transforms
from datasets import load_dataset
from torch.utils.data import DataLoader
from my_awesome_app.my_strategy import CustomFedAvg


def get_evaluate_fn(testloader, device):
    """Return a callback that evaluate the global model"""

    def evaluate(server_round, parameters_ndarray, config):
        """Evaluate global model using provided centralised testset"""
        net = Net()
        set_weights(net, parameters_ndarray)
        net.to(device)
        loss, accuracy = test(net, testloader, device)
        return loss, {"cen_accuracy": accuracy}

    return evaluate


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """callback: how to aggregate metrics sent back from the clients app that evaluate locally the global model into strategy/Server (weight by number of samples)"""
    # for each metric in the metrics client list
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    return {"accuracy": sum(accuracies) / total_examples}  # can add also other metrics


# def handle_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
#     """handle metrics and clients."""
#     # is called at the end of every client fit round
#     # iterates over list of (client_id, metrics) tuples -> aggregate clients metrics (max)
#     b_values = []
#     for _, m in metrics:
#         my_metric_str = m["my_metric"]
#         my_metric = json.loads(my_metric_str)
#         b_values.append(my_metric["b"])
#
#     return {"max_b": max(b_values)}


def on_fit_config(server_round: int) -> Metrics:
    """" adjust learning rate based on the server round """
    initial_lr = 0.005
    decay_factor = 0.95
    min_lr = 0.0001  # Prevent underfitting

    lr = max(initial_lr * math.pow(decay_factor, server_round), min_lr)
    # saved to client config (used in fit method)
    return {"lr": lr}


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(
        Net())  # could load check points model here (global_model_round_1) to resume training on last global model
    parameters = ndarrays_to_parameters(ndarrays)

    # load global test set
    testset = load_dataset("uoft-cs/cifar10")["test"]
    # preprocessing image data (transform to tensor(hugging dataset) + normalize pixel values)
    testloader = DataLoader(testset.with_transform(get_transforms()), batch_size=32)

    # Define strategy
    # strategy = FedAvg(
    strategy = CustomFedAvg(
        fraction_fit=fraction_fit,
        # Fraction (0.5) of clients to perform fit in each round -> 100% is too expensive, prevent overfitting, prevent poisons attacks (not consistently integrated)
        fraction_evaluate=1.0,  # Fraction of clients used during validation
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,  # optional, server metrics weighted aggregation function
        on_fit_config_fn=on_fit_config,  # Function used to configure training (learning rate)
        evaluate_fn=get_evaluate_fn(testloader, device="cpu"),  # Optional, function used for validation
        #  fit_metrics_aggregation_fn=handle_fit_metrics,  # optional, Metrics aggregation function
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
