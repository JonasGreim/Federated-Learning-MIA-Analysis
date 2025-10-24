import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flower.utils.data_loading import load_data_custom
from flower.utils.model_utils import get_weights, set_weights
from flower.utils.reproducibility import seed_everything
from flower.utils.training import test, train
from path_settings import D3_SPLIT_PATH, D1_SPLIT_PATH
from flower.utils.model_factory import create_model
import time

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, device):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = device
        self.net.to(self.device)

    def fit(self, parameters, config):
        t0 = time.perf_counter()
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
        print(f"fit: {time.perf_counter() - t0:.3f}s", flush=True)
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
    batch_size = context.run_config.get("batch-size", 32)
    seed = context.run_config.get("seed", 42)
    model_name = context.run_config.get("model")
    iid = context.run_config.get("iid-data-distribution", True)
    alpha = context.run_config.get("dirichlet-alpha", 1.0)
    train_target_model_as_shadow_model = context.run_config.get("train-target-model-as-shadow-model", False)

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

    t0 = time.perf_counter()
    trainloader, valloader = load_data_custom(iid=iid, dirichlet_alpha=alpha, partition_id=partition_id,
                                              num_partitions=num_partitions, split=split,
                                              batch_size=batch_size, seed=seed)
    print(f"loader_build={time.perf_counter() - t0:.3f}s", flush=True)

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs, device).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
