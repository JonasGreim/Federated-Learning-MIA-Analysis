from flwr.common import FitRes, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
import torch
import json
import wandb
from datetime import datetime
from .task import Net, set_weights


#  extends FedAvg and overrides some methods to add custom behavior, such as saving the global model and logging metrics.
class CustomFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.result_to_save = {}  # empty dictionary to save results

        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wandb.init(project="flower-simulation-tutorial", name=f"custom-strategy-{name}")

    def aggregate_fit(
            self,
            server_round: int,
            results: list[tuple[ClientProxy, FitRes]],
            failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        # convert to ndarrays (n-dimensional arrays)
        ndarrays = parameters_to_ndarrays(parameters_aggregated)

        # instantiate model (Pytorch way)
        model = Net()
        set_weights(model, ndarrays)

        # save global model in the standard PyTorch way
        torch.save(model.state_dict(), f"global_model_round_{server_round}")

        return parameters_aggregated, metrics_aggregated

    # override evaluate method to test global model + save metrics and log them to W&B
    # custom strategy evaluation (evaluation in server app like default evaluation)
    def evaluate(
            self, server_round: int, parameters: Parameters
    ) -> tuple[float, dict[str, Scalar]] | None:
        loss, metrics = super().evaluate(server_round, parameters)

        my_result = {"loss": loss, **metrics}

        self.result_to_save[server_round] = my_result

        # save metrics as json
        with open("results.json", 'w') as json_file:
            json.dump(self.result_to_save, json_file, indent=4)

        # log to W&B (also json metrics)
        wandb.log(my_result, step=server_round)

        return loss, metrics
