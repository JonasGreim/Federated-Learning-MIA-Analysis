"""my-awesome-app: A Flower / PyTorch app."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from my_awesome_app.task import get_weights, set_weights, test, train, create_model, load_data_custom, \
    get_dataset_split_flag, seed_everything
import numpy as np


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, device):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = device
        self.net.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train_loss, train_accuracy = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            config['lr'],
            self.device,
        )

        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss, "train_accuracy": train_accuracy},  # "my_metric": complex_metrix_str
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"evaluate_accuracy": accuracy, "evaluate_loss": loss}


def client_fn(context: Context):
    # Load model and data
    net = create_model()

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    local_epochs = context.run_config.get("local-epochs", 1)
    device_str = context.run_config.get("device", "cpu")
    batch_size = context.run_config.get("batch_size", 32)
    seed = context.run_config.get("seed", 42)

    # Seed everything for reproducibility
    seed_everything(seed)

    # check if cuda is available and set device accordingly
    device = torch.device(device_str if torch.cuda.is_available() or "cpu" in device_str else "cpu")
    print(f"[Client {partition_id}] Using device: {device}, gpu available {torch.cuda.is_available()}")

    split_type = get_dataset_split_flag()
    if split_type == "target":
        indices = np.load("splits/D1_indices.npy").tolist()
    elif split_type == "shadow":
        indices = np.load("splits/D3_indices.npy").tolist()
    else:
        raise ValueError(f"Unknown dataset-split: {split_type}")

    trainloader, valloader = load_data_custom(partition_id=partition_id, num_partitions=num_partitions, indices=indices,
                                              batch_size=batch_size, seed=seed)

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs, device).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
