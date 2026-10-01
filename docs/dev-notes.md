# Developer notes

This document preserves useful operational and architectural details that are too specific for the project overview. It reflects the current repository; planned improvements are labeled explicitly.

## Flower runtime architecture

The FL implementation uses Flower's application and deployment components:

- `ServerApp` in `flower/server_app.py` initializes the global model, strategy, dataset visualization, centralized evaluation, and server rounds.
- `ClientApp` in `flower/client_app.py` loads one partition, performs local training/evaluation, and returns parameters and metrics.
- `FedCustom` in `flower/strategies/custom_weighted_fedavg.py` implements weighted aggregation, centralized evaluation, overfitting metrics, checkpoints, local JSON output, and W&B logging.
- In deployed mode, SuperLink coordinates the run and exchanges tasks with one SuperNode per client node. Flower handles transport between these processes.
- In simulation mode, Ray resources and the number of SuperNodes come from the federation definitions in `pyproject.toml`.

The Hydra `flower.num_clients` field is used for visualization and naming; it does not allocate simulation SuperNodes. Keep it consistent with the actual deployment when producing label-distribution plots.

## Configuration layers

Two configuration layers coexist:

1. `pyproject.toml` defines the Flower app, direct-Flower defaults, and local/deployed federation resources.
2. `experiments_conf/config.yaml` composes Hydra FL and MIA groups for the Python wrappers.

`run_flower_experiment.py` translates underscore-style Hydra keys into Flower's hyphenated run-config keys and starts the selected local federation. `run_flower_hpc.py` performs the equivalent handoff to `hpc-deploy`.

Experiment files are nested below subgroup directories. Select them with their full path, for example:

```bash
python3 run_flower_experiment.py flower=main_mia_experiments/run1 flower.device=cpu
python3 run_mia_experiment.py mia=main_mia_experiments/run1 mia.parameters_static.device=cpu
```

Existing values can be overridden without changing the dataclasses. Update `experiments_conf_types/` only when a field or type changes.

## Checkpoint lifecycle and FL-to-MIA handoff

At FL startup, `create_model_checkpoint_save_folder` creates the next free numeric subdirectory below `CHECKPOINTS_DIR_TARGET`.

- Checkpoints are saved every fifth round and at the final round.
- The final checkpoint is uploaded as a W&B artifact.
- The numeric folder is not derived from the Hydra or W&B run name.
- The MIA does not discover a checkpoint automatically.

An MIA config must set:

- `mia.parameters.target_model_folder`
- `mia.parameters.target_model_file`
- `mia.parameters.model_arch`

The architecture is instantiated before the state dictionary is loaded, so it must match the checkpoint. Shadow-training settings are configured independently and should be kept consistent with the intended threat model.

## Data generation and replacement

`flower/utils/split_cifar10_mia.py` creates the four persistent Hugging Face datasets. `ensure_split_data_exists()` regenerates them only when at least one D1-D4 directory is missing; it does not detect changed ratios.

After intentionally changing split ratios, move or remove the old generated splits before the next run. Dataset directories are ignored by Git.

Replacing CIFAR-10 is not a config-only operation. Review at least:

- the Hugging Face dataset identifier and column adapter
- split construction and class balance assumptions
- transforms and image channels/dimensions
- target and shadow architectures
- `num_classes` and `class_names` in the MIA config
- analysis labels and plots

## Runtime outputs

The current output locations are intentionally separate:

- Checkpoints: `CHECKPOINTS_DIR_TARGET/<numeric-run>/global_model_round_<round>.pth`
- FL metrics: `METRICS_DIR/flower/<run-name>__<timestamp>/federated_learning_rounds.json`
- MIA metrics: `METRICS_DIR/mia/<run-name>__<timestamp>/mia_per_class.json` and `mia_summary.json`
- Runtime plots: the same timestamped metric directory
- Hydra logs: `outputs/<date>/<time>/`
- Selected config snapshots: `outputs/${exp_name}/flower_config.yaml` or `mia_config.yaml`
- Local W&B files: `wandb/`
- Aggregated thesis artifacts: `experiments_data_analysis/data/`, `figures/`, and `metrics/`

The metric-folder timestamp is precise to one minute and `exist_ok=True` is used. Repeating the same run name within one minute can reuse a folder.

Analysis scripts that download W&B data use entity/project names and paths from `experiments_data_analysis/file_name_settings.py`.

## PAULA and SLURM logging

The scripts in `hpc_slurm_scripts/` reproduce the thesis deployment and are environment-specific.

`flower.sbatch` writes scheduler output to `logs/flower_<job-id>.out` and `.err`. It also creates `logs/flower_<job-id>/` for:

- `superlink.log`
- one `supernode_<node>.log` per client node
- one `flwr_run<id>.log` per Hydra experiment

`mia.sbatch` writes scheduler output to `logs/mia_<array-job-id>_<task-id>.out` and `.err`. Per-run command output is also tee'd into `logs/mia_<array-job-id>/run<id>.log`.

The deployment scripts assume PAULA-specific partitions, node names/exclusions, ports, Conda environment, project directory, and shared `/work` storage. `flower.sbatch` temporarily replaces the `hpc-deploy` server address in `pyproject.toml` and restores its placeholder after the run.

## Known implementation constraints

These are documented rather than fixed by the README task:

- `path_settings.py` activates repository-local paths. Using the commented PAULA `/work` layout still requires a deliberate source edit; runtime roots are not selected through Hydra or environment variables.
- W&B initialization and logging are unconditional in FL and MIA code.
- `run_flower_experiment.py` compares a `torch.device` object with the string `"cuda"` when selecting a federation. With the audited environment this selects the CPU federation even when CUDA was requested.
- `run_multiple_experiments.py` builds the obsolete selector `flower=run0`; current configs require a subgroup-qualified path.
- Stored MIA experiment configs assume that corresponding numeric checkpoint folders already exist.
- Several analysis labels and generated plots remain in German.
- The repository declares Apache-2.0 package metadata but does not contain a `LICENSE` file.

## Planned engineering improvements

These ideas were retained from the old README as possible future engineering work; they are not implemented commitments:

- Replace numeric checkpoint folders with stable run metadata or an explicit FL-to-MIA manifest.
- Make runtime roots deploy-time configurable through Hydra or environment variables instead of editing `path_settings.py` for PAULA.
- Add a tested W&B offline/disabled execution mode.
- Consolidate experiment-grid launchers and validate Hydra selectors before starting work.
- Split `membership_inference_attack/mia_shokri.py` into smaller training, feature, attack, and evaluation modules.
- Use shared data interfaces and plotting style configuration in runtime and analysis code.
- Standardize generated labels and plot text in English.
- Add alternative black-box feature sets or attack classifiers only as separately evaluated experiments.

Research extensions supported by the thesis are summarized in the main README rather than here.
