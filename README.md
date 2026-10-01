# Membership Inference Attacks on Federated Learning

**How do model architecture and training choices affect membership leakage from a federated global model?**

This repository implements a reproducible experimental pipeline that combines federated learning (FL) with a **black-box shadow-model Membership Inference Attack (MIA) based on Shokri et al. (2017)**. It applies this established attack to examine how model architecture, client-data heterogeneity, local training, regularization, and overfitting relate to membership leakage from the final global model. The project does not introduce a new MIA method.

The project was developed for the 2026 Master's thesis *Einfluss von Trainingsparametern und Modellarchitekturen auf die Verwundbarkeit von Federated Learning gegenüber Membership-Inference-Angriffen* at Leipzig University. The thesis is written in German.

<!-- TODO: Add a public link to the German thesis PDF. -->

## Table of contents

- [Motivation](#motivation)
- [Key findings](#key-findings)
- [Running the project](#running-the-project)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Experiment configuration](#experiment-configuration)
  - [Running Federated Learning](#running-federated-learning)
  - [Running the Membership Inference Attack](#running-the-membership-inference-attack)
  - [Running the complete FL-to-MIA pipeline](#running-the-complete-fl-to-mia-pipeline)
- [Experimental setup](#experimental-setup)
- [How the Membership Inference Attack works](#how-the-membership-inference-attack-works)
- [Project structure](#project-structure)
- [Extending and configuring the project](#extending-and-configuring-the-project)
- [Scope and limitations](#scope-and-limitations)
- [References](#references)

## Motivation

Federated learning keeps raw training data on participating clients, but the resulting model can still reveal information about its training set. A membership inference attack asks whether a particular sample was used during training.

This project evaluates that risk for global FL models and examines whether common training choices and measured overfitting help explain the observed attack strength.

## Key findings

The main study evaluated 36 FL configurations: two architectures, three client data distributions, three local-epoch settings, and training with or without regularization.

- **The attack achieved above-random performance in all 36 evaluated configurations:** MIA-AUC ranged from **0.57 to 0.81**.
- **Model architecture was the strongest observed factor:** ResNet-18 reached **0.74–0.81** AUC (median about **0.78**), compared with **0.57–0.67** (median about **0.63**) for Shokri-CNN.
- **Regularization had architecture-dependent effects:** median AUC changed from **0.64 to 0.60** for Shokri-CNN and from **0.79 to 0.77** for ResNet-18.
- **Overfitting was positively related to attack strength, but did not fully explain it:** the relationship was much weaker within a fixed architecture, and models with similar overfitting gaps could have different attack AUCs.

<!-- TODO: Add thesis Figure 5.2 (architecture comparison) here. -->

### Illustrative heterogeneity result

The following configurations use one local epoch per communication round and no regularization:

| Model | Client data distribution | Overfitting-Gap-Loss | MIA-AUC |
| --- | --- | ---: | ---: |
| Shokri-CNN | IID | 1.01 | 0.66 |
| Shokri-CNN | semi-non-IID (alpha = 0.5) | 1.19 | 0.63 |
| Shokri-CNN | non-IID (alpha = 0.1) | 1.40 | 0.65 |
| ResNet-18 | IID | 2.12 | 0.81 |
| ResNet-18 | semi-non-IID (alpha = 0.5) | 2.45 | 0.80 |
| ResNet-18 | non-IID (alpha = 0.1) | 3.09 | 0.79 |

In this example, greater client-data heterogeneity coincides with a larger overfitting gap, while MIA-AUC varies by at most 0.03. This pattern is illustrative rather than universal: other evaluated configurations behave differently.

### Validation experiments

- **Number of clients:** for 2, 5, and 10 clients, Shokri-CNN AUC was **0.71, 0.66, and 0.55**, respectively, while ResNet-18 AUC was **0.81, 0.82, and 0.81**, respectively, in the evaluated setup.
- **Number of shadow models:** testing **1, 3, 10, and 20** shadow models showed no systematic improvement from adding more models. A single shadow model was sufficient in the evaluated setup, where the attacker knew the target architecture and training procedure.

## Running the project

Repository-local execution is the default: [`path_settings.py`](path_settings.py) places generated data splits, checkpoints, and metrics below the repository root. A fresh clone still needs its dependencies, W&B access, and a first-run CIFAR-10 download before it can execute an experiment. The default experiments are compute-intensive rather than smoke tests.

### Requirements

- Python **3.10–3.12** (`>=3.10,<3.13`)
- Poetry for the lock-file-based installation shown below
- Access to Hugging Face on first dataset generation
- Weights & Biases credentials for online execution
- Sufficient compute resources for FL training or the default 100-epoch shadow-model training

W&B is part of the current execution path: the FL and MIA implementations initialize a run and log metrics unconditionally. If the machine is not already authenticated for online use, run `poetry run wandb login` before starting an experiment.

### Installation

From the repository root:

```bash
poetry env use python3.12
poetry install
```

If Python 3.12 is unavailable, select another interpreter in the supported 3.10–3.12 range. Run the entry points through Poetry as shown below, or activate the Poetry environment first.

### Experiment configuration

Hydra composes [`experiments_conf/config.yaml`](experiments_conf/config.yaml) with the selected FL and MIA configuration groups:

- FL defaults: [`experiments_conf/flower/base.yaml`](experiments_conf/flower/base.yaml); experiment groups: [`experiments_conf/flower/`](experiments_conf/flower/)
- MIA defaults: [`experiments_conf/mia/base.yaml`](experiments_conf/mia/base.yaml); experiment groups: [`experiments_conf/mia/`](experiments_conf/mia/)
- Configuration dataclasses: [`experiments_conf_types/`](experiments_conf_types/)
- Flower application defaults and simulation resources: [`pyproject.toml`](pyproject.toml)

Use fully qualified CLI overrides such as `flower.device=cpu`, `mia.parameters.num_shadow_models=1`, or `mia.parameters.target_model_file=...`. Stored experiment configurations are nested below subgroups, so selectors must include the subgroup—for example, `flower=main_mia_experiments/run1` or `mia=main_mia_experiments/run1`. Direct `flwr run` commands use the separate defaults and federation definitions in `pyproject.toml`; the Python FL wrappers inject the selected Hydra values into Flower's run configuration.

The local defaults in `path_settings.py` are:

- dataset splits: `dataset_splits/D1` through `dataset_splits/D4`
- target checkpoints: `model_checkpoints_target/`
- FL and MIA metrics: `metrics/flower/` and `metrics/mia/`

If any D1–D4 split is missing, the application downloads CIFAR-10 from Hugging Face and generates all four splits. Generated `dataset_splits/`, `metrics/`, `outputs/`, and `wandb/` directories are ignored by Git, as are numeric checkpoint subdirectories; the packaged `model_checkpoints_target/example/` directory is tracked.

### Running Federated Learning

For a local CPU simulation using the base Hydra configuration:

```bash
poetry run python run_flower_experiment.py flower.device=cpu
```

The base Hydra configuration trains for 100 rounds. FL checkpoints are saved every fifth round and at the final round. Each run creates the next available numeric directory under `model_checkpoints_target/`; the selected folder is printed when the run starts. Local simulation SuperNode count and CPU/GPU resource allocation come from the federation definitions in `pyproject.toml`, not from Hydra's `flower.num_clients` field.

### Running the Membership Inference Attack

The tracked checkpoint [`model_checkpoints_target/example/global_model_round_10.pth`](model_checkpoints_target/example/global_model_round_10.pth) loads as the configured Shokri-CNN (`simple_model`). Its other training provenance is intentionally not claimed here.

Run the default MIA against that checkpoint with:

```bash
poetry run python run_mia_experiment.py
```

This is the simplest included example, but it is not a lightweight smoke test: the default configuration trains one shadow model for 100 epochs on 15,000 training images. It requests CUDA and falls back to CPU when CUDA is unavailable. To request CPU explicitly:

```bash
poetry run python run_mia_experiment.py mia.parameters_static.device=cpu
```

### Running the complete FL-to-MIA pipeline

First run FL and note the numeric checkpoint directory printed at startup:

```bash
poetry run python run_flower_experiment.py flower.device=cpu
```

The MIA does not discover the newest FL checkpoint automatically. Replace `RUN_FOLDER` below with the numeric directory created by FL. With the unchanged base FL configuration, its final checkpoint is `global_model_round_100.pth` and its architecture is `simple_model`:

```bash
poetry run python run_mia_experiment.py \
  mia.parameters.target_model_folder=RUN_FOLDER \
  mia.parameters.target_model_file=global_model_round_100.pth \
  mia.parameters.model_arch=simple_model \
  mia.parameters_static.device=cpu
```

The target checkpoint and `model_arch` must be compatible. Stored MIA configurations reference numeric checkpoint directories from the thesis experiments; those artifacts must exist under the active checkpoint root, or the folder and filename must be overridden.

### Outputs

Runtime and analysis artifacts are split across several repository-local locations:

- target checkpoints: `model_checkpoints_target/<numeric-run>/global_model_round_<round>.pth`
- FL and MIA metrics and plots: timestamped run directories under `metrics/flower/` and `metrics/mia/`
- selected Hydra sections: `outputs/${exp_name}/flower_config.yaml` or `mia_config.yaml`
- Hydra run logs: date/time directories below `outputs/`
- local W&B run data: `wandb/`
- aggregated thesis data, figures, and tables: `experiments_data_analysis/`

`path_settings.py` is the authoritative source for the split, checkpoint, and metric roots.

### HPC / PAULA execution

The multi-node path uses [`run_flower_hpc.py`](run_flower_hpc.py) and [`hpc_slurm_scripts/`](hpc_slurm_scripts/). These scripts capture the PAULA deployment used for the thesis and contain environment-specific project paths, Conda setup, node settings, ports, and scheduler assumptions; they are not generic SLURM launchers.

The active `path_settings.py` values also apply during HPC execution unless deliberately changed. A commented PAULA `/work` configuration is retained in that file for reference, but it is not the default.

## Experimental setup

### Federated learning system

- Horizontal, synchronous FL implemented with Flower
- FedAvg aggregation
- Five clients in the main study, with all clients participating in every round
- CIFAR-10 dataset
- Shokri-CNN and ResNet-18 target architectures
- IID, semi-non-IID (Dirichlet alpha = 0.5), and non-IID (alpha = 0.1) client distributions
- One, five, or ten local epochs per communication round
- Either no regularization or weight decay `5e-4` combined with dropout `0.5`
- Fixed seeds and persistent metrics

Every client completed 100 local epochs in total. Communication rounds were adjusted to keep this total constant:

| Local epochs per round | Communication rounds | Total local epochs per client |
| ---: | ---: | ---: |
| 1 | 100 | 100 |
| 5 | 20 | 100 |
| 10 | 10 | 100 |

The thesis experiments ran FL on six PAULA nodes—one server and five clients—and ran each MIA on one node. Each node provided four NVIDIA Tesla A30 GPUs with 24 GB memory and used Rocky Linux 9.

### Disjoint data partitions

| Split | Size | Purpose |
| --- | ---: | --- |
| D1 | 20,000 | FL target-model training; known members during MIA evaluation |
| D2 | 10,000 | Global-model evaluation; known non-members during MIA evaluation |
| D3 | 15,000 | Shadow-model training pool |
| D4 | 15,000 | Shadow-model test pool |

D2 is the original CIFAR-10 test set. D1, D3, and D4 form disjoint partitions of the original training set. Each client partition is split again into 80% local training data and 20% local evaluation data.

### Overfitting indicator

The study uses the **Overfitting-Gap-Loss (OGL)**, comparing aggregated client-side training loss with the global model's server-side test loss. OGL is treated as an empirical indicator of generalization behavior, not as a direct measurement of privacy leakage.

## How the Membership Inference Attack works

The attack follows the shadow-model approach of Shokri et al. (2017) and targets the final aggregated global model.

### Threat model

The attacker is passive and operates through black-box queries. It can observe the target model's softmax prediction probabilities and is assumed to know the architecture and training procedure. It has separate data from the same underlying distribution.

The attacker does **not** receive:

- target-model parameters or gradients
- local client models or intermediate updates
- client training data
- access to or influence over the federated training process

```text
CIFAR-10
   |
   +-- D1 --> federated target-model training --> final global model
   |                                              |
   |                                              +-- D1 members ----+
   |                                              +-- D2 non-members +--> evaluation
   |
   +-- D3/D4 --> central shadow model(s)
                    |
                    +--> member/non-member features
                              |
                              +--> one logistic-regression model per class
```

### Attack pipeline

1. Train one or more central shadow models on D3.
2. Query each shadow model with its training samples (members) and D4 samples (non-members).
3. Combine the full softmax vector with a one-hot encoding of the true class.
4. Pool features by class across shadow models.
5. Balance member and non-member samples and standardize the features.
6. Train one binary logistic-regression attack model for each CIFAR-10 class.
7. Query the federated target model with D1 members and D2 non-members.
8. Evaluate membership predictions with the corresponding class-specific attack model.

MIA-AUC is the primary metric. The implementation also records accuracy, precision, recall, F1, and false-acceptance rate.

## Project structure

```text
.
├── flower/                         # Flower server/client apps, models, strategy, data utilities
├── membership_inference_attack/    # Shadow-model MIA and MIA logging
├── experiments_conf/               # Hydra FL and MIA configurations
├── experiments_conf_types/         # Configuration dataclasses
├── experiments_creation_scritps/   # Experiment-config generation scripts
├── experiments_data_analysis/      # Aggregated data, analysis, figures, and tables
├── hpc_slurm_scripts/               # PAULA/SLURM deployment scripts
├── model_checkpoints_target/        # Target checkpoints; tracked example subdirectory
├── docs/                            # Stable project documentation and audit
├── run_flower_experiment.py         # Local FL simulation wrapper
├── run_flower_hpc.py                # Deployed Flower run wrapper
├── run_mia_experiment.py            # Membership inference entry point
├── path_settings.py                 # Dataset, checkpoint, and metric roots
└── pyproject.toml                   # Package metadata and Flower configuration
```

Generated `dataset_splits/`, `metrics/`, `outputs/`, and `wandb/` directories may also appear locally but are not versioned. Numeric checkpoint directories are likewise ignored; only the packaged `model_checkpoints_target/example/` directory is tracked.

## Extending and configuring the project

- **Add a model:** implement it in [`flower/models/`](flower/models/) and register its config name in [`flower/utils/model_factory.py`](flower/utils/model_factory.py). Both FL and MIA instantiate architectures through this factory.
- **Change aggregation behavior:** adapt [`flower/strategies/custom_weighted_fedavg.py`](flower/strategies/custom_weighted_fedavg.py), which currently handles aggregation, centralized evaluation, overfitting metrics, checkpoints, JSON output, and W&B logging.
- **Adjust experiments:** extend the YAML groups under [`experiments_conf/flower/`](experiments_conf/flower/) and [`experiments_conf/mia/`](experiments_conf/mia/). Add dataclass fields under [`experiments_conf_types/`](experiments_conf_types/) only when the configuration schema changes.
- **Change data splits:** see [`flower/utils/split_cifar10_mia.py`](flower/utils/split_cifar10_mia.py). Existing generated splits must be regenerated after ratio changes.
- **Use another dataset:** this is not a config-only extension. The dataset source, transforms/adapter, split logic, input dimensions, models, class count, and MIA class names all require review.
- **Rebuild analysis artifacts:** analysis-side W&B project names and output paths are in [`experiments_data_analysis/file_name_settings.py`](experiments_data_analysis/file_name_settings.py).

Research facts are curated in [`docs/thesis-facts.md`](docs/thesis-facts.md). Operational details and known implementation constraints are documented in [`docs/dev-notes.md`](docs/dev-notes.md), while verified README claims and their sources are recorded in [`docs/readme-audit.md`](docs/readme-audit.md).

## Scope and limitations

- Results apply to the evaluated setup: CIFAR-10, two architectures, FedAvg, and five clients in the main experiment.
- The attack targets the final aggregated global model under a passive black-box threat model.
- White-box attacks, active attackers, attacks on individual client models, and intermediate-round attack dynamics were not evaluated.
- Shadow models were trained centrally; the attacker was assumed to know the target architecture and training procedure.
- The analysis is correlational and does not establish causal effects.
- A single shadow model was sufficient in the evaluated setup.
- No formal privacy guarantees or dedicated defenses such as differential privacy were evaluated.

### Next steps from the thesis

- Train shadow models through federated learning and evaluate stronger or active attackers.
- Study client sampling and alternative aggregation methods such as FedProx and SCAFFOLD.
- Vary model capacity systematically through depth and width.
- Analyze overfitting and attack signals across communication rounds.
- Evaluate defenses such as differential privacy.

## References

1. R. Shokri, M. Stronati, C. Song, and V. Shmatikov. *Membership Inference Attacks Against Machine Learning Models.* IEEE Symposium on Security and Privacy, 2017.
2. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas. *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS, 2017.
3. S. Yeom, I. Giacomelli, M. Fredrikson, and S. Jha. *Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting.* IEEE Computer Security Foundations Symposium, 2018.
4. A. Salem, Y. Zhang, M. Humbert, P. Berrang, M. Fritz, and M. Backes. *ML-Leaks: Model and Data Independent Membership Inference Attacks and Defenses on Machine Learning Models.* NDSS, 2019.
