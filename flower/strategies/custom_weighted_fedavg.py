from typing import Union

from datasets import load_from_disk
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from typing import Optional, List, Tuple, Dict
from flower.utils.data_loading import get_transforms_custom
from flower.utils.huggingface_to_pytorch import HFDatasetToTorch
from torch.utils.data import DataLoader
import json
import torch
from flower.utils.model_factory import create_model
from flower.utils.training import test
from flower.utils.model_utils import set_weights
from flower.utils.reproducibility import release_model
from flower.utils.wandb_logging import wandb_log_metrics
from path_settings import (
    CHECKPOINTS_DIR_TARGET,
    CHECKPOINTS_DIR_SHADOW,
    D2_SPLIT_PATH,
    D4_SPLIT_PATH,
    METRICS_DIR,
    ensure_dir_exist,
)


class FedCustom(Strategy):
    def __init__(
            # parameters with default values
            self,
            fraction_fit: float = 0.5,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            initial_parameters: Optional[Parameters] = None,
            learning_rate: float = 0.001,
            lr_decay: float = 0.0,
            weight_decay: float = 0.0,
            train_target_model_as_shadow_model: bool = False,
            device: str = "cpu",
            model_name: str = "mia_paper",
            batch_size: int = 32,
    ) -> None:
        # can be overwritten here + abstract methods invoke
        super().__init__()
        if "cuda" in device and not torch.cuda.is_available():
            print("[FedCustom] ⚠️ CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.initial_parameters = initial_parameters
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.train_target_model_as_shadow_model = train_target_model_as_shadow_model
        self.result_to_json_global_model_test = {}
        self.device = torch.device(device)
        self.model_name = model_name
        self.batch_size = batch_size

    def __repr__(self) -> str:
        return "FedCustom"

    def initialize_parameters(
            self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def configure_fit(
            self,
            server_round: int,
            parameters: Parameters,
            client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training with a fixed learning rate."""

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # spread fit parameters and config to clients
        config = {"lr": self.learning_rate, "lr_decay": self.lr_decay, "weight_decay": self.weight_decay}
        return [(client, FitIns(parameters, config)) for client in clients]

    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        loss_aggregated = weighted_loss_avg(
            [
                (fit_res.num_examples, fit_res.metrics.get("train_loss", 0.0))
                for _, fit_res in results
            ]
        )

        accuracy_aggregated = weighted_loss_avg(  # same functionality as loss
            [
                (fit_res.num_examples, fit_res.metrics.get("train_accuracy", 0.0))
                for _, fit_res in results
            ]
        )
        metrics_aggregated = {"client_weighted_train_loss": loss_aggregated,
                              "client_weighted_train_accuracy": accuracy_aggregated}
        wandb_log_metrics(metrics=metrics_aggregated, step=server_round)

        aggregated_ndarrays = aggregate(weights_results)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        #  save global model each round
        model = create_model(self.model_name)
        set_weights(model, aggregated_ndarrays)

        # save global model from shadow and target model in the standard PyTorch way
        if self.train_target_model_as_shadow_model:
            save_dir = CHECKPOINTS_DIR_SHADOW
        else:
            save_dir = CHECKPOINTS_DIR_TARGET

        ensure_dir_exist(save_dir)
        model_path = save_dir / f"global_model_round_{server_round}.pth"
        torch.save(model.state_dict(), model_path)
        release_model(model, self.device.type)
        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""

        if not results:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        accuracy_aggregated = weighted_loss_avg(  # same functionality as loss
            [
                (evaluate_res.num_examples, evaluate_res.metrics.get("evaluate_accuracy", 0.0))
                for _, evaluate_res in results
            ]
        )

        metrics_aggregated = {"client_weighted_evaluate_loss": loss_aggregated,
                              "client_weighted_evaluate_accuracy": accuracy_aggregated}

        wandb_log_metrics(metrics=metrics_aggregated, step=server_round)

        return loss_aggregated, metrics_aggregated

    def evaluate(
            self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""

        if self.train_target_model_as_shadow_model:
            split = D4_SPLIT_PATH
        else:
            split = D2_SPLIT_PATH

        try:
            test_split_dataset = load_from_disk(split)
        except Exception as e:
            raise RuntimeError(f"Target model: Failed to load test dataset from disk: {e}") from e

        testset = HFDatasetToTorch(test_split_dataset, transform=get_transforms_custom())

        testloader = DataLoader(testset, batch_size=self.batch_size)

        net = create_model(self.model_name)
        set_weights(net, parameters_to_ndarrays(parameters))  # parameters = tensors -> ndarray parameters
        net.to(self.device)
        loss, accuracy = test(net, testloader, self.device)

        # log results to json and wandb
        result = {"cen_loss": loss, "cen_accuracy": accuracy}
        self.result_to_json_global_model_test[server_round] = result

        # save metrics as json
        ensure_dir_exist(METRICS_DIR)
        with open(METRICS_DIR / "results.json", "w") as json_file:
            json.dump(self.result_to_json_global_model_test, json_file, indent=4)

        # log to W&B (also json metrics)
        wandb_log_metrics(metrics=result, step=server_round)
        release_model(net, self.device.type)
        return loss, result

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients
