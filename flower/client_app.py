"""my-awesome-app: A Flower / PyTorch app."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flower.utils.task import get_weights, set_weights, test, train, create_model, load_data_custom, seed_everything
from path_settings import D3_SPLIT_PATH, D1_SPLIT_PATH


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
            config['lr_decay'],
            config['weight_decay'],
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
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    local_epochs = context.run_config.get("local-epochs", 1)
    device_str = context.run_config.get("device", "cpu")
    batch_size = context.run_config.get("batch_size", 32)
    seed = context.run_config.get("seed", 42)
    model_name = context.run_config.get("model")
    iid = context.run_config.get("iid_data_distribution", True)
    alpha = context.run_config.get("dirichlet_alpha", 1.0)
    train_target_model_as_shadow_model = context.run_config.get("train_target_model_as_shadow_model", False)

    # Seed everything for reproducibility
    seed_everything(seed)

    # check if cuda is available and set device accordingly
    device = torch.device(device_str if torch.cuda.is_available() or "cpu" in device_str else "cpu")
    print(f"[Client {partition_id}] Using device: {device}, gpu available {torch.cuda.is_available()}")

    # Load model and data
    net = create_model(model_name)

    if train_target_model_as_shadow_model:
        split = D3_SPLIT_PATH
    else:
        split = D1_SPLIT_PATH

    trainloader, valloader = load_data_custom(iid=iid, dirichlet_alpha=alpha, partition_id=partition_id, num_partitions=num_partitions, split=split,
                                              batch_size=batch_size, seed=seed)

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs, device).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
