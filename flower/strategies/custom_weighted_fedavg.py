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
from flower.utils.model_utils import set_weights, create_save_folder
from flower.utils.reproducibility import release_model
from flower.utils.wandb_logging import wandb_log_metrics, wandb_upload_artifact_model
from path_settings import (
    CHECKPOINTS_DIR_TARGET,
    D2_SPLIT_PATH,
    D4_SPLIT_PATH,
    ensure_dir_exist, METRICS_DIR_FLOWER,
)
from pathlib import Path


class FedCustom(Strategy):
    def __init__(
            # parameters with default values
            self,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            initial_parameters: Optional[Parameters] = None,
            learning_rate: float = 0.01,
            lr_decay: float = 0.0,
            weight_decay: float = 0.0,
            train_target_model_as_shadow_model: bool = False,
            device: str = "cpu",
            model_name: str = "simple_model",
            batch_size: int = 32,
            max_server_rounds: int = 10,
            metric_save_folder: Path = METRICS_DIR_FLOWER,
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
        self.device = torch.device(device)
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_server_rounds = max_server_rounds

        self.model_saving_folder = create_save_folder(save_dir=CHECKPOINTS_DIR_TARGET)
        print(f"Created model checkpoint folder: {self.model_saving_folder}")
        self.all_round_metrics = {}
        self.cache_metric = {}
        self.metric_save_folder = metric_save_folder

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
        metrics_aggregated = {"Trainings-Loss (Clients aggregiert)": loss_aggregated,
                              "Trainings-Accuracy (Clients aggregiert)": accuracy_aggregated}
        self.cache_metric.update(metrics_aggregated)
        wandb_log_metrics(metrics=metrics_aggregated, step=server_round)

        aggregated_ndarrays = aggregate(weights_results)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        #  save global model each round
        model = create_model(self.model_name)
        set_weights(model, aggregated_ndarrays)

        # save global model from shadow and target model in the standard PyTorch way
        ensure_dir_exist(self.model_saving_folder)
        model_path = self.model_saving_folder / f"global_model_round_{server_round}.pth"

        # save model every 5 rounds or at the last round
        if server_round % 5 == 0 or server_round == self.max_server_rounds:
            torch.save(model.state_dict(), model_path)

        # Upload the final model to W&B as an artifact
        if self.max_server_rounds == server_round:
            wandb_upload_artifact_model(artifact_name=f"{self.model_name}-{server_round}", artifact_path=model_path)

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

        validation_loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        validation_accuracy_aggregated = weighted_loss_avg(  # same functionality as loss
            [
                (evaluate_res.num_examples, evaluate_res.metrics.get("evaluate_accuracy", 0.0))
                for _, evaluate_res in results
            ]
        )

        metrics_aggregated = {"Validierungs-Loss (Clients aggregiert)": validation_loss_aggregated,
                              "Validierungs-Accuracy (Clients aggregiert)": validation_accuracy_aggregated,
                              }
        self.cache_metric.update(metrics_aggregated)
        wandb_log_metrics(metrics=metrics_aggregated, step=server_round)

        return validation_loss_aggregated, metrics_aggregated

    def _compute_overfitting_gaps(self, server_loss, server_accuracy):
        """Compute overfitting gaps based on cached metrics."""
        train_loss_aggregated = self.cache_metric.get("Trainings-Loss (Clients aggregiert)")
        train_accuracy_aggregated = self.cache_metric.get("Trainings-Accuracy (Clients aggregiert)")
        validation_loss_aggregated = self.cache_metric.get("Validierungs-Loss (Clients aggregiert)")
        validation_accuracy_aggregated = self.cache_metric.get("Validierungs-Accuracy (Clients aggregiert)")

        # calculate overfitting gap if possible (first server round there is no validation)
        client_overfitting_gap_loss = (
                validation_loss_aggregated - train_loss_aggregated) if validation_loss_aggregated is not None and train_loss_aggregated is not None else None
        client_overfitting_gap_accuracy = (
                train_accuracy_aggregated - validation_accuracy_aggregated) if train_accuracy_aggregated is not None and validation_accuracy_aggregated is not None else None
        server_overfitting_gap_loss = (
                    server_loss - train_loss_aggregated) if train_loss_aggregated is not None else None
        server_overfitting_gap_accuracy = (
                    train_accuracy_aggregated - server_accuracy) if train_accuracy_aggregated is not None else None

        return {"Overfitting-Gap (Clients aggregiert, Loss)": client_overfitting_gap_loss,
                "Overfitting-Gap (Clients aggregiert, Accuracy)": client_overfitting_gap_accuracy,
                "Overfitting-Gap (Server, Loss)": server_overfitting_gap_loss,
                "Overfitting-Gap (Server, Accuracy)": server_overfitting_gap_accuracy
                }

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
        server_loss, server_accuracy = test(net, testloader, self.device)

        metrics_overfitting_gaps = self._compute_overfitting_gaps(server_loss=server_loss, server_accuracy=server_accuracy)

        server_metrics = {"Test-Loss (Server)": server_loss, "Test-Accuracy (Server)": server_accuracy, **metrics_overfitting_gaps}

        # save metrics as json
        all_metrics = {**server_metrics, **self.cache_metric}
        self.all_round_metrics[server_round] = all_metrics
        with open(self.metric_save_folder / "results.json", "w") as json_file:
            json.dump(self.all_round_metrics, json_file, indent=4)

        # log to W&B
        wandb_log_metrics(metrics=server_metrics, step=server_round)

        release_model(net, self.device.type)
        return server_loss, server_metrics

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients
