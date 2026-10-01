# README audit (Phase 1)

Date: 2026-09-30

This audit compares the current implementation and configuration with the final-thesis fact sheet, the final thesis where the fact sheet was incomplete, the old `README.md`, and the two candidate drafts. The README and drafts were treated as claims to verify, not as sources of truth. No FL/MIA experiment, SLURM job, network download, or W&B login/upload was run.

Maintenance update: `path_settings.py` was subsequently changed to activate repository-local split, checkpoint, and metric roots. The affected findings below were re-checked against the current file on 2026-09-30; current implementation behavior supersedes the original path audit.

## Classification

The `README disposition` column implements the four requested categories:

1. **Include** - verified and useful in the main README.
2. **Separate docs** - verified, but too implementation-specific for the main README.
3. **Reject/update** - outdated or contradicted; do not repeat as written.
4. **Unverifiable** - not established by the available authoritative sources.

`Status` uses the task's required vocabulary: **verified**, **contradicted**, or **unverifiable**. A verified implementation detail may still be category 2, and a verified statement that another document is wrong is category 3.

## Claim audit

### Installation, packaging, license, and external services

| Claim | Source | Status | README disposition | Note |
| --- | --- | --- | --- | --- |
| Supported Python is `>=3.10,<3.13` (Python 3.10-3.12). | `pyproject.toml` `[project].requires-python` | verified | 1 - Include | This is the authoritative range. The drafts' 3.10-3.12 wording is correct after removing the unverified "developed with 3.12" claim. |
| Dependencies are declared with PEP 621; Hatchling is the build backend. | `pyproject.toml` `[project]`, `[build-system]` | verified | 1 - Include | Main runtime stack: Flower simulation, Ray, PyTorch/torchvision, Hugging Face Datasets/Flower Datasets, scikit-learn, Hydra/OmegaConf, W&B, NumPy, pandas, matplotlib, seaborn, and Pillow. Keep the README summary short and link to `pyproject.toml`. |
| Poetry is the repository's only possible package manager. | `pyproject.toml`; `poetry.lock` header; `poetry check` | contradicted | 3 - Reject/update | A Poetry 2.1.4 lock file exists and `poetry check` passes, so `poetry install` is a supported reproducible workflow. The project itself uses standard PEP 621 metadata and a Hatchling backend, so "requires Poetry" is too strong. A clean install was not executed because it could access package indexes and modify the environment. |
| Installation can be documented as `poetry env use python3.12` followed by `poetry install`. | `poetry.lock`; `pyproject.toml`; successful `poetry check` | verified | 1 - Include | State Poetry as the lock-file workflow, not as an application runtime requirement. A generic `pip install -e .` should be possible from the build metadata, but it was not tested and would not use `poetry.lock`; do not promise it in Phase 2 without a clean-environment check. |
| The current clone URL is `https://github.com/JonasGreim/Federated-Learning-MIA-Analysis.git`. | Both drafts; Git remote `git@github.com:JonasGreim/Flower2.git` | contradicted | 3 - Reject/update | The checked-out repository's origin is `JonasGreim/Flower2`; the proposed public/renamed URL is not verifiable. Use a repository-relative install sequence or a confirmed final URL. |
| The project is licensed Apache-2.0. | `pyproject.toml` `[project].license` | verified | 1 - Include with caveat | Package metadata declares `Apache-2.0`. No `LICENSE`, `LICENSE.*`, or `COPYING*` file exists, so the repository does not currently contain the license text. Do not present licensing as fully resolved. |
| A LICENSE file is present. | Repository file inventory | contradicted | 3 - Reject/update | No license file exists. The task already leaves adding one out of scope. |
| `masterarbeit.pdf` is available in this working tree but excluded from Git. | `.gitignore`; `git ls-files masterarbeit.pdf`; repository inventory | verified | 1 - Include as placeholder only | The PDF is ignored and untracked, as required because it contains personal data. A README link cannot target the local file in a fresh clone; retain the requested TODO until a deliberate public thesis URL exists. |
| W&B is optional in the current application code. | `flower/utils/wandb_logging.py`; `flower/server_app.py`; `flower/strategies/custom_weighted_fedavg.py`; `membership_inference_attack/mia_shokri.py`; `membership_inference_attack/utils/wandb_logging_mia.py` | contradicted | 3 - Reject/update | W&B is imported and `wandb.init`, `wandb.log`, summary updates, file saves, and final-model artifact upload are called unconditionally. There is no repository config flag or no-op logger. |
| `wandb login` is mandatory before every run. | Same W&B sources; `hpc_slurm_scripts/flower.sbatch` | contradicted | 3 - Reject/update | Online runs need credentials, but an already authenticated environment does not need another login. W&B also supports externally selected offline/disabled modes, although the repository does not expose or document one. README should say W&B integration is mandatory in code, while interactive login is only needed for unauthenticated online use. Offline behavior was not executed. |
| All important results are also written locally, so W&B can simply be omitted. | Strategy and MIA logging sources above | contradicted | 3 - Reject/update | FL round metrics and MIA summaries are written locally, but several plots/tables and all tracking calls still go through a live W&B run object. Local persistence does not make the W&B integration optional. |

### Entry points, defaults, Hydra, and run behavior

| Claim | Source | Status | README disposition | Note |
| --- | --- | --- | --- | --- |
| Main developer entry points are `run_flower_experiment.py`, `run_mia_experiment.py`, and `run_flower_hpc.py`. | Those files; `pyproject.toml` Flower components | verified | 1 - Include | The first wraps local Flower simulation, the second runs the MIA, and the third targets the deployed `hpc-deploy` federation. There are no `[project.scripts]` console entry points. Direct `flwr run . [federation]` is also available through the Flower app configuration. |
| `run_multiple_experiments.py` is a ready-to-use experiment-grid runner. | `run_multiple_experiments.py`; config inventory | contradicted | 3 - Reject/update | It currently runs only index 0 and constructs `flower=run0`, which is not a valid current Hydra group path. Keep it out of the main README unless repaired. |
| With no override, Hydra composes `experiments_conf/config.yaml` + `flower/base.yaml` + `mia/base.yaml`. | Those files; static `--cfg job` checks | verified | 1 - Include | Both FL and MIA sections are present in the composed config even when only one entry point uses one section. `exp_name` defaults to `default`; config snapshots go to `outputs/default/flower_config.yaml` or `outputs/default/mia_config.yaml`. |
| `python3 run_mia_experiment.py mia=run1` and `python3 run_flower_experiment.py flower=run1` select run 1. | Drafts and old README vs. config tree; static Hydra check | contradicted | 3 - Reject/update | Hydra reports `Could not find 'mia/run1'`. Current files are nested. Valid selectors include `mia=main_mia_experiments/run1` and `flower=main_mia_experiments/run1`; analogous subgroup paths are required for client-count and shadow-model configs. |
| Individual values can be overridden from the CLI. | Hydra composition; static `--cfg job` checks | verified | 1 - Include | Use the fully qualified key, e.g. `flower.device=cpu` or `mia.parameters_static.device=cpu`. For checkpoint selection use `mia.parameters.target_model_folder=...` and `mia.parameters.target_model_file=...`. YAML files do not need to be edited for one-off overrides. |
| Draft-listed FL keys exist: `num_server_rounds`, `local_epochs`, `weight_decay`, `model`, `iid_data_distribution`, `dirichlet_alpha`, and `num_clients`. | `experiments_conf/flower/base.yaml`; `experiments_conf_types/types_config_flower.py` | verified | 2 - Separate docs | They are spelled exactly this way in Hydra YAML/dataclasses; the wrapper translates them to hyphenated Flower run-config keys. Link to YAML in the README rather than copying defaults. |
| Draft-listed MIA concepts have config keys for shadow count/architecture/epochs, train/test sizes, weight decay, run name, and checkpoint folder/file. | `experiments_conf/mia/base.yaml`; `experiments_conf_types/types_config_mia.py` | verified | 2 - Separate docs | Keys live under `mia.parameters.*`; device, seed, batch size, learning rates, worker count, and class names live under `mia.parameters_static.*`. |
| Changing any YAML value requires changing the dataclass. | Old README vs. `experiments_conf_types/` | contradicted | 3 - Reject/update | Existing values can be overridden freely. Dataclasses need adjustment only when the schema/field set or type changes, and Hydra is not configured with a registered structured-config schema that validates values at composition time. |
| `pyproject.toml` and Hydra share one set of defaults. | `pyproject.toml`; `experiments_conf/flower/base.yaml`; `run_flower_experiment.py` | contradicted | 3 - Reject/update | There are two layers. Direct `flwr run .` uses TOML defaults (for example 10 rounds, batch size 32, nominal `num-clients=2`). The Python wrapper composes Hydra defaults (100 rounds, batch size 128, nominal `num_clients=5`) and injects most values through `--run-config`. Explain this distinction. |
| The Hydra `num_clients` value determines the number of local simulation clients. | `experiments_conf/flower/base.yaml`; `pyproject.toml`; `flower/server_app.py` | contradicted | 3 - Reject/update | Source comments say this value is used only for visualization/naming. Local simulation uses `options.num-supernodes=4` in the selected federation; deployed HPC gets node count from SuperNodes/SLURM. The default Hydra value 5 therefore does not make the default local simulation a five-client run. |
| Default device is CUDA for FL and MIA, with CPU fallback when CUDA is unavailable. | Both base YAML files; `run_flower_experiment.py`; `flower/client_app.py`; `FedCustom.__init__`; `mia_shokri.run_mia` | verified | 1 - Include | Both YAML defaults are `cuda`; the wrapper, clients, strategy, and MIA contain fallback logic. Direct overrides are preferable to editing YAML. |
| The local wrapper reliably chooses the GPU federation when CUDA is selected. | `run_flower_experiment.py`; read-only runtime check of `torch.device('cuda') == 'cuda'` | contradicted | 3 - Reject/update | The wrapper compares a `torch.device` object to the string `"cuda"`; this is false with the installed PyTorch 2.8.0, so it selects `local-simulation-cpu` even while injecting `device="cuda"`. Do not claim the wrapper's GPU-resource selection works until fixed/tested. Direct `flwr run . local-simulation-gpu` selects the GPU federation defined in TOML. |
| `experiments_conf/flower/base.yaml` defines Flower client CPU/GPU resource allocation. | Base YAML; `pyproject.toml` federation tables | contradicted | 3 - Reject/update | The device request is in YAML; Ray client resources (`num-cpus`, `num-gpus`) are in the TOML federation definitions. Claude draft line 158 conflates them. |
| Local/simulation and multi-node/HPC execution paths both exist. | `run_flower_experiment.py`; `run_flower_hpc.py`; `pyproject.toml`; `hpc_slurm_scripts/`; `path_settings.py` | verified | 1 - Include, qualified | Both are implemented. Repository-local paths are now active by default, so local mode does not require path reconfiguration. It still requires installed dependencies, W&B access, and a first-run dataset download. Deployment scripts remain PAULA-specific, and the active paths remain repository-local during HPC execution unless deliberately changed. |
| The HPC scripts are generic SLURM scripts. | `hpc_slurm_scripts/*.sbatch`, `*.sh`; `path_settings.py`; `pyproject.toml` | contradicted | 3 - Reject/update | They hard-code the `paula` partition, project/conda paths, node selections/exclusions, ports, shared storage paths, and mutate the deployment address in `pyproject.toml` during a job. They are evidence of a working thesis deployment, not a portable turnkey interface. |

### MIA implementation and target-checkpoint handoff

| Claim | Source | Status | README disposition | Note |
| --- | --- | --- | --- | --- |
| The thesis attack uses one binary logistic-regression classifier per CIFAR-10 class. | `docs/thesis-facts.md` pp. 29-31; `masterarbeit.pdf` pp. 29-31 | verified | 1 - Include | Features are the full softmax vector plus one-hot true label; features are standardized and member/non-member examples are balanced per class. |
| Current code implements logistic regression per class. | `membership_inference_attack/mia_shokri.py` lines 235-294 | verified | 1 - Include | `LogisticRegression(max_iter=1000, random_state=seed)` is active. A RandomForest line is commented out. Code and thesis agree. |
| The old README's attack classifier is a `RandomForestClassifier`. | Old `README.md` line 192 vs. current MIA code and thesis facts | contradicted | 3 - Reject/update | This is obsolete. The nearby source comment saying RandomForest is "better" is not evidence of an evaluated result and belongs neither in the README nor the scientific conclusions. |
| The default MIA uses one shadow model. | `experiments_conf/mia/base.yaml`; static composed config; thesis facts pp. 29-31 | verified | 1 - Include | Main study also used one. Validation configs use 1, 3, 10, and 20. |
| Default MIA values are 10 classes, 1 shadow model, Shokri-CNN (`simple_model`), 100 shadow epochs, 15,000 train + 15,000 test examples, no weight decay, batch size 128, seed 42, 6 workers, CUDA, and example round-10 checkpoint. | `experiments_conf/mia/base.yaml`; `experiments_conf_types/types_config_mia.py`; static config composition | verified | 2 - Separate docs | In the main README, link to the YAML instead of duplicating all defaults, per the Phase 2 target structure. Mention only the values necessary to understand the example and study. |
| More than one shadow model always uses the entire D3 pool per model. | Shadow-model validation YAMLs; `sample_shadow_datasets_from_mia_data_pool`; thesis facts pp. 59-60 | contradicted | 3 - Reject/update | One model uses 15,000 examples; the 3/10/20-model validation configs sample 10,000 per model, without replacement within each subset but with overlap across shadow models. |
| The MIA automatically attacks the checkpoint produced by the preceding FL command. | `run_mia_experiment.py`; `mia_shokri.run_mia`; MIA YAMLs | contradicted | 3 - Reject/update | There is no automatic handoff or "latest checkpoint" discovery. MIA resolves `CHECKPOINTS_DIR_TARGET / target_model_folder / target_model_file`. The user/config must name the correct folder and file. |
| FL run folders are named after the Hydra/W&B run name. | `flower/utils/model_utils.py` | contradicted | 3 - Reject/update | Checkpoint folders are the next available numeric directory (`0`, `1`, ...). The run name is not used. This makes the FL-to-MIA mapping non-obvious and should be called out. |
| Model architecture is inferred from checkpoint contents. | `mia_shokri.load_specific_target_model`; MIA config | contradicted | 3 - Reject/update | `parameters.model_arch` creates the model first; the state dict is then loaded strictly. Architecture must match manually. Other training settings such as weight decay are also independently configured for shadow training. |
| Research conclusions concern the final global checkpoint, while code can target another saved round. | Thesis facts pp. 29-31; `mia_shokri.py`; `custom_weighted_fedavg.py` | verified | 1 - Include | Main experiment configs point to final-round files. The implementation accepts any named compatible checkpoint; do not generalize thesis results to intermediate rounds. |
| The included round-10 example checkpoint exists and matches `simple_model`. | `model_checkpoints_target/example/global_model_round_10.pth`; `experiments_conf/mia/base.yaml`; read-only strict state-dict load | verified | 1 - Include | Both round 5 and round 10 files are tracked. The default MIA selects round 10. |
| The example checkpoint was trained for 10 server rounds with 10 local epochs and IID data. | Old README and Claude draft; checkpoint filename/state dict; repository metadata | unverifiable | 4 - Unverifiable | The filename establishes round 10, and the architecture is load-verifiable, but no tracked config/provenance file links this example directory to 10 local epochs or IID data. Do not repeat those training settings without provenance. |
| `python3 run_mia_experiment.py` is an out-of-the-box, offline quick start from a fresh clone. | Drafts/old README vs. `path_settings.py`, `.gitignore`, config, source | contradicted | 3 - Reject/update | The repository-local checkpoint path is now active and the included checkpoint is valid. A fresh clone still needs installed dependencies, W&B access, and a Hugging Face download because `dataset_splits/` is ignored/not tracked. The default shadow training is also 100 epochs on 15,000 images, so the command is the simplest included example but not a fast or offline smoke test. |
| FL checkpoints are saved every fifth round. | `custom_weighted_fedavg.py` lines 171-181 | verified | 1 - Include, corrected | They are saved every fifth round **and at the final round**, including when the final round is not divisible by five. Only the final checkpoint is uploaded as a W&B artifact. |
| FL training must precede every MIA run. | Included checkpoint; default MIA config | contradicted | 3 - Reject/update | MIA needs a compatible target checkpoint, not necessarily a newly trained one. The included example can avoid FL training once paths/prerequisites are configured. For new experiments: run FL, identify its numeric checkpoint directory, then select that exact checkpoint in the MIA config/CLI. |

### Data, outputs, repository layout, and extension points

| Claim | Source | Status | README disposition | Note |
| --- | --- | --- | --- | --- |
| CIFAR-10 is split into disjoint D1=20,000, D2=10,000, D3=15,000, D4=15,000 with the documented target/shadow roles. | `docs/thesis-facts.md` pp. 13-17, 24, 29-31; `flower/utils/split_cifar10_mia.py` | verified | 1 - Include | D2 is the original CIFAR-10 test set; D1/D3/D4 partition the original train set. Client partitions are further split 80/20 locally. |
| Dataset splits ship with a fresh clone. | `.gitignore`; `git ls-files`; current working tree | contradicted | 3 - Reject/update | The local working tree contains them, but `dataset_splits/` is ignored and no split files are tracked. `ensure_split_data_exists()` downloads/generates all splits when any are missing, requiring network access to Hugging Face. |
| Split sizes can be changed only by editing `flower/utils/split_cifar10_mia.py`; existing splits then need regeneration. | That file; `ensure_split_data_exists()` | verified | 2 - Separate docs | The code only checks presence, not whether ratios changed. The old splits must be removed/moved before regeneration. This is an advanced, potentially destructive workflow; explain it outside the main README. |
| Adding an implemented model requires adding it under `flower/models/` and registering it in `flower/utils/model_factory.py`. | Model directory and factory | verified | 1 - Include | Config strings must match a factory branch. Both target and shadow paths call the same factory, but config consistency is not enforced. |
| Alternative aggregation is an extension point in `flower/strategies/custom_weighted_fedavg.py`. | Strategy and `flower/server_app.py` | verified | 1 - Include, qualified | The current `FedCustom` implements aggregation, centralized evaluation, checkpointing, JSON metrics, overfitting gaps, and W&B calls. Replacing it is a code change, not a config-only plug-in. |
| Another dataset can be enabled by changing only split constants. | Split, loading, model, and MIA sources | contradicted | 3 - Reject/update | CIFAR-10 assumptions occur in the Hugging Face dataset name, image column adapter/transforms, 10 classes/class names, input dimensions, split semantics, model architectures, and MIA config. Keep a detailed migration checklist in separate docs. |
| `file_name_settings.py` exists at repository root. | Repository inventory | contradicted | 3 - Reject/update | It exists at `experiments_data_analysis/file_name_settings.py`. It configures analysis-side W&B entity/project names and output paths; it does not control training/MIA runtime paths. |
| `path_settings.py` controls checkpoint, split, and metric directories and defines output/W&B directory constants. | `path_settings.py`; run and logging sources | verified | 1 - Include, qualified | Split, checkpoint, and metric roots are now repository-local by default. `OUTPUTS_DIR` and `WANDB_DIR` are defined but are not used by the inspected run/logging code: Hydra uses its own output convention and W&B is initialized with `ROOT_DIR`. |
| Checkpoints are written under repository `model_checkpoints_target/`. | Current `path_settings.py`; checkpoint helper and strategy | verified | 1 - Include | The active checkpoint root is `ROOT_DIR / "model_checkpoints_target"`. The tracked `example/` checkpoint is available to the default MIA, while generated numeric run folders are ignored. |
| FL and MIA metrics are written as JSON under `metrics/`. | Strategy/MIA loggers; current `path_settings.py` | verified | 1 - Include | Active destinations are repository-local `metrics/flower` and `metrics/mia`, each using timestamped per-run folders. PNG plots are also produced. |
| Hydra logs and composed-config snapshots share one output location. | Hydra observed `outputs/YYYY-MM-DD/HH-MM-SS/*.log`; run wrappers; `experiments_conf/config.yaml` | contradicted | 3 - Reject/update | Hydra creates date/time run logs under `outputs/`; the wrappers separately save the selected section to `outputs/${exp_name}/..._config.yaml`. Document them as distinct outputs. |
| W&B local run files are written under repository `wandb/`. | `initialize_wandb_run(..., dir=ROOT_DIR)`; observed ignored `wandb/run-*` files | verified | 2 - Separate docs | Useful for troubleshooting; too detailed for the main project narrative. |
| Analysis data, aggregate figures/tables, and downloaded per-run metrics live under `experiments_data_analysis/`. | Directory inventory; `experiments_data_analysis/file_name_settings.py` | verified | 1 - Include | Distinguish these thesis analysis artifacts from runtime `metrics/`, `outputs/`, and `wandb/`. Some analysis download scripts require W&B access. |
| Important directory responsibilities in the drafts are broadly correct. | Repository inventory | verified | 1 - Include | Main README should cover `flower/`, `membership_inference_attack/`, `experiments_conf/`, `experiments_conf_types/`, `experiments_data_analysis/`, `hpc_slurm_scripts/`, and the run scripts. Also mention generated/ignored `dataset_splits/`, checkpoint, metrics, outputs, and W&B directories without implying they all ship in a clone. |
| `experiments_creation_scritps/` is misspelled but is the actual directory. | Repository inventory | verified | 2 - Separate docs | Preserve the real path in links. Avoid highlighting the typo in the portfolio overview unless users need the generator scripts. |

### Research and institutional claims

| Claim | Source | Status | README disposition | Note |
| --- | --- | --- | --- | --- |
| The study design and all numerical findings in the drafts match the curated fact sheet. | `docs/thesis-facts.md` (pages cited there) | verified | 1 - Include | The 36 configurations, AUC ranges/medians, heterogeneity table, client-count results, shadow-count result, data splits, rounds table, and limitations are supported. Preserve the cautious "in the evaluated setup" and correlational wording. |
| "Model architecture is the dominant factor." | Claude draft vs. approved thesis-fact wording | contradicted | 3 - Reject/update | Use "Model architecture was the strongest observed factor." "Dominant" is more absolute than the approved wording. |
| The attack proves privacy leakage or causal parameter effects. | Thesis facts pp. 42-50 | contradicted | 3 - Reject/update | The evaluation shows above-random membership inference in the evaluated setup and correlations, not causation or universal leakage. |
| Thesis runs used PAULA, with six nodes for FL and one for MIA. | `masterarbeit.pdf` p. 26; `hpc_slurm_scripts/`; recorded in `docs/thesis-facts.md` | verified | 1 - Include if desired | The PDF explicitly names PAULA and provides the hardware/runtime facts. The scripts also use `#SBATCH --partition=paula`. |
| The public README may name PAULA and Leipzig University. | Technical sources above; drafts | unverifiable | 4 - Unverifiable | Sources can establish technical provenance, not permission or portfolio preference. "Leipzig University" is asserted in the ChatGPT draft and encoded indirectly in an analysis W&B entity, but the text-extracted thesis cover names the institute/faculty/group without spelling out the university name. The owner should explicitly approve both names before Phase 2. |

## Contradictions and outdated material to resolve

The highest-impact conflicts are:

1. **Attack classifier:** thesis and current code use per-class logistic regression; the old README says RandomForest.
2. **Quick start:** the default MIA now resolves the included checkpoint through repository-local paths, but it still requires dependency installation, W&B access, first-run dataset generation, and substantial compute. It is not an out-of-the-box offline smoke test.
3. **Hydra selectors:** all three README variants use `mia=run1` / `flower=run1`; the current configuration tree requires subgroup-qualified selectors such as `mia=main_mia_experiments/run1`.
4. **FL-to-MIA handoff:** drafts imply a direct two-command pipeline; code creates the next numeric checkpoint directory and requires the MIA config to name it manually.
5. **Defaults:** direct Flower runs use TOML values while wrapper runs inject Hydra values. The drafts present only one default set.
6. **Client count/resources:** Hydra `num_clients` is visualization metadata, while actual local SuperNodes/resources live in TOML. The default values do not agree (5 vs. 4 simulated SuperNodes; TOML's nominal app config says 2).
7. **GPU simulation:** a device-object/string comparison makes the local wrapper select the CPU federation even for CUDA, so the current GPU claim needs qualification or a code fix outside this task.
8. **Output locations:** repository-local checkpoints and metrics are now the active defaults. Hydra logs, config snapshots, metrics, checkpoints, W&B files, and analysis artifacts still remain separate output classes.
9. **Portable HPC support:** multi-node code exists, but scripts are PAULA-specific and not generic cluster instructions.
10. **Old setup/history:** old README sections describing 10 clients, 5 rounds, `fraction-fit=0.5`, `my-awesome-app`, 4+ shadow models, and the missing `shadow_models_central_per_class_true_shokri.py` command are obsolete.
11. **Clone URL:** candidate drafts name a repository URL different from the checked-out Git origin.
12. **License:** metadata says Apache-2.0, but no license text is present.

No contradiction was found between the thesis fact sheet and current code for the core evaluated MIA classifier, data split sizes/roles, main one-shadow-model setup, or the 100-total-local-epoch rounds scheme. The implementation is more general than the evaluated design in some places, so the README must keep research claims separate from configurable code behavior.

## Developer-facing facts missing from both drafts

These should be documented before calling the README quick start complete:

- `path_settings.py` now uses repository-local split, checkpoint, and metric roots by default; PAULA storage is a commented, opt-in configuration.
- `dataset_splits/` is ignored and absent from a fresh clone. First generation downloads CIFAR-10 from Hugging Face, so it requires network access and disk space.
- W&B calls are unconditional. Explain credentials for online mode and a verified offline workflow if offline use is intended.
- Hydra experiment selectors need their subgroup (`main_mia_experiments/...`, `client_number_mia_experiments/...`, or `shadow_models_experiments/...`).
- Direct key overrides are nested (`flower.*`, `mia.parameters.*`, `mia.parameters_static.*`).
- The TOML Flower defaults and Hydra wrapper defaults are different configuration layers.
- Actual local simulation client count/resources come from the TOML federation, not Hydra `num_clients`.
- The local wrapper's current CUDA/federation selection defect should be fixed or documented before advertising GPU simulation.
- FL creates a new numeric checkpoint folder. MIA does not discover it; users must select the folder/file explicitly and keep `model_arch` compatible.
- The default MIA trains a 100-epoch shadow model on 15,000 images. It is the simplest available example, but not a lightweight smoke test.
- Checkpoints are saved every five rounds and at the final round; only the final checkpoint is uploaded as a W&B artifact.
- Runtime outputs are split across Hydra logs, saved Hydra section configs, local metrics/plots, checkpoints, W&B local files, SLURM logs, and analysis artifacts.
- `run_multiple_experiments.py` is not currently usable with the nested config layout.
- No console scripts are installed; commands run repository-root Python files or the Flower CLI.
- The packaged example checkpoint is tracked, but most generated checkpoints, metrics, outputs, W&B runs, and dataset splits are ignored.

## What belongs in the main README

- Python range, Poetry lock-file install workflow, and a concise dependency/stack summary.
- W&B's actual runtime role and the authentication/offline prerequisite, without claiming it is optional.
- The three useful entry points and correct Hydra subgroup/override syntax.
- A corrected example-checkpoint workflow, including first-run prerequisites and the fact that it trains a shadow model.
- The explicit FL -> numeric checkpoint -> MIA checkpoint-selection relationship.
- CUDA defaults, CPU fallback/override, and a concise warning that simulation resources are configured separately in `pyproject.toml`.
- Conditional wording that local simulation and a PAULA-specific multi-node deployment implementation exist.
- Where each major class of output is intended to go, with a link to `path_settings.py`.
- High-level directory responsibilities and the verified model/strategy extension points.
- Research design/results/limitations exactly as supported by `docs/thesis-facts.md`, with correlational wording.

## What should move to separate developer documentation

- Exact TOML-versus-Hydra precedence and a complete key reference.
- PAULA SLURM topology, ports, node selection, Conda environment, log naming, and deployment-address mutation.
- How numeric checkpoint folders are allocated and how to map a completed FL run to a MIA config safely.
- Complete output-directory and retention/cleanup behavior.
- Dataset replacement and split-regeneration checklists.
- Analysis download/rebuild workflow and `experiments_data_analysis/file_name_settings.py`.
- W&B online/offline/disabled operational recipes after they have been tested.
- Troubleshooting CPU worker/pinned-memory behavior and Ray resource allocation.
- Experiment-config generation scripts and `run_multiple_experiments.py` limitations.
- The old README's Flower architecture notes, HPC log details, historical TODOs, and "what could be done better" material that Phase 2 requires preserving in `docs/dev-notes.md` when still useful.

## Unverifiable claims

- That the project was specifically "developed with Python 3.12" (supported range is verified; development history is not).
- That the proposed `Federated-Learning-MIA-Analysis` GitHub URL exists or will be the final public URL.
- That PAULA and Leipzig University should be publicly named; this is an owner/editorial decision even where technical provenance is available.
- That a generic pip installation or W&B offline run is currently end-to-end functional; neither was executed in this audit.
- That local or PAULA execution currently succeeds end-to-end after dependency installation; full experiments and SLURM jobs were intentionally not run.
- That the current code has been tested on every Python version in the declared 3.10-3.12 range.

## Phase 2 decisions

- `docs/` is the canonical location for stable documentation.
- PAULA and Leipzig University may be named where relevant.
- No hard-coded clone URL is used because the draft URL is not verified as the final public repository name.
- The README omits a License section until a license file exists.
- Repository-local path defaults are documented separately from the optional PAULA `/work` layout; further path configurability remains outside this task.
- W&B is described as part of the current execution path, not as optional.
- The compute-heavy default MIA is documented honestly rather than presented as a smoke test.
- Lower-level implementation defects remain in this audit and `docs/dev-notes.md` unless they directly affect a documented path.

## Verification performed and intentionally skipped

Performed:

- Inspected current source, YAML/TOML config, run scripts, SLURM scripts, output helpers, repository inventory, Git tracking/ignore state, old README, and both drafts.
- Ran `poetry check` successfully.
- Used Hydra `--cfg job` only (no experiment) to verify default composition, valid subgroup selectors, and nested overrides; confirmed `mia=run1` fails.
- Loaded the included round-10 checkpoint on CPU and strictly matched it to `simple_model`; no training or inference was run.
- Checked installed PyTorch device equality behavior without starting Flower.
- Used the final thesis only for the missing PAULA wording/page fact, then recorded it in `docs/thesis-facts.md` per `AGENTS.md`.

Skipped:

- Clean dependency installation, dataset download, FL/MIA execution, GPU workload, W&B initialization/login/upload, and all SLURM jobs.
- End-to-end validation of local/HPC execution, because the task explicitly prohibits full experiments, external services, and SLURM jobs.
