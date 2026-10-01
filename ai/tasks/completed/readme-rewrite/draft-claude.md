# Membership Inference Attacks on Federated Learning

**How much do model architecture and training parameters influence the privacy leakage of a federated global model?**

An empirical study of black-box *membership inference attacks* (MIA) against global models trained with Federated Learning (FL). Built with PyTorch and [Flower](https://flower.ai/), configured with Hydra, tracked with Weights & Biases, and run on an HPC cluster via SLURM.

This repository contains the code of my Master's thesis (Computer Science, Data Privacy and Security group, 2026):
*"Einfluss von Trainingsparametern und Modellarchitekturen auf die Verwundbarkeit von Federated Learning gegenüber Membership-Inference-Angriffen"* (written in German).
<!-- TODO: add link to the PDF, e.g. [Read the thesis (PDF, German)](docs/thesis.pdf) -->

---

## Motivation

Federated Learning keeps raw training data on the clients, but this does not guarantee that the resulting global model is free from privacy leakage. A membership inference attack asks: **was this specific data point part of the training set?** Even this yes/no answer can be sensitive, for example when the data comes from hospitals.

For centrally trained models, overfitting is a known risk factor for such attacks. It is much less clear whether this carries over to FL, where the global model emerges from many local training runs and server-side aggregation.

## Key findings

The study covers **36 FL configurations** (2 architectures × 3 data distributions × 3 local-epoch settings × with/without regularization). One attack was run against each.

- **The attack beat random guessing in all 36 configurations**, with an attack AUC between **0.57 and 0.81**.
- **Model architecture is the dominant factor.** ResNet-18 was consistently easier to attack (AUC 0.74–0.81, median ≈ 0.78) than the shallow Shokri-CNN (AUC 0.57–0.67, median ≈ 0.63).
- **Training parameters have smaller, architecture-dependent effects.** Data distribution, local epochs and regularization changed the attack AUC only moderately and often non-monotonically. For example, regularization lowered the median AUC from 0.64 to 0.60 for the Shokri-CNN, but only from 0.79 to 0.77 for ResNet-18.
- **Overfitting is a useful but incomplete indicator.** The overfitting gap correlates positively with attack strength across all configurations, but the relationship is much weaker within a fixed architecture, and models with similar overfitting gaps can differ strongly in attack AUC.

<!-- TODO: add your best figure, e.g. the architecture comparison (thesis Fig. 5.2):
![Attack AUC by model architecture](experiments_data_analysis/figures/<your_figure>.png)
-->

### Example: more data heterogeneity, more overfitting, but not more leakage

No regularization, 1 local epoch per round. Data heterogeneity is controlled via a Dirichlet distribution (IID, α = 0.5, α = 0.1).

| Model | Data distribution | Overfitting gap (OGL) | Attack AUC |
|---|---|---|---|
| Shokri-CNN | IID | 1.01 | 0.66 |
| Shokri-CNN | semi-non-IID (α = 0.5) | 1.19 | 0.63 |
| Shokri-CNN | non-IID (α = 0.1) | 1.40 | 0.65 |
| ResNet-18 | IID | 2.12 | 0.81 |
| ResNet-18 | semi-non-IID (α = 0.5) | 2.45 | 0.80 |
| ResNet-18 | non-IID (α = 0.1) | 3.09 | 0.79 |

The overfitting gap grows with heterogeneity, while the attack AUC changes by at most 0.03. (This trend depends on the setting: for ResNet-18 with 10 local epochs the gap does not grow.) The full table for all 36 configurations is in the thesis (Table 5.1); aggregated data and figures are in [`experiments_data_analysis/`](experiments_data_analysis).

### Validation experiments

- **Number of clients (2, 5, 10):** for the Shokri-CNN, attack AUC dropped from 0.71 to 0.66 to 0.55, together with a much lower overfitting gap. For ResNet-18 it stayed at 0.81–0.82.
- **Number of shadow models (1, 3, 10, 20):** more shadow models did not systematically improve the attack in the evaluated setup, so the main experiments use a single one. This holds for an attacker who knows the target architecture and training procedure.

---

## Experimental setup

**Federated system**
- Horizontal, synchronous FL with FedAvg. The main experiments use 5 clients that all participate in every round.
- Two architectures: Shokri-CNN (small) and ResNet-18 (deeper, higher capacity).
- Every configuration trains for 100 local epochs per client in total. The number of communication rounds is adjusted to the local epochs per round (1, 5 or 10), so only the *intensity* of local training changes.
- Regularization variant: weight decay (5·10⁻⁴) plus dropout (0.5).
- Runs as a local simulation or as a real multi-node deployment. The thesis experiments used 6 nodes (1 server, 5 clients) on a SLURM cluster with NVIDIA A30 GPUs.
- Random seeds are fixed and all metrics are stored persistently.

**Data splits (CIFAR-10, strictly disjoint)**

| Split | Size | Purpose |
|---|---|---|
| D1 | 20,000 | FL training data, used as **members** |
| D2 | 10,000 | Evaluation of the global model, used as **non-members** (from the original test set) |
| D3 | 15,000 | Shadow model training |
| D4 | 15,000 | Shadow model testing |

**Overfitting indicator:** Overfitting-Gap-Loss (OGL), the difference between the aggregated client-side training loss and the server-side test loss of the global model.

## How the attack works

The attack follows the shadow-model approach of [Shokri et al. (2017)](https://arxiv.org/abs/1610.05820), applied to the final global model of an FL run. The attacker is passive and works in a black-box setting: it can query the model and see softmax probabilities, but has **no** access to model parameters, gradients, client models, client data or the federated training process.

```text
                          CIFAR-10
                             |
                 disjoint partitions D1 - D4
              +--------------+---------------+
              |                              |
        D1 (members)                    D3 / D4
              |                              |
              v                              v
   FL training, 5 clients (FedAvg)     shadow model(s), trained centrally
              |                              |
              v                              v
     global target model            softmax outputs, labeled
              |                       member / non-member
              |                              |
              |                              v
              |                    per-class attack models
              |                              |
              +--------------+---------------+
                             |
                             v
        query target model with D1 (members) and D2 (non-members)
                             |
                             v
              attack evaluation (ROC-AUC, F1, ...)
```

1. Train one or more shadow models on data disjoint from the target's training data, with the same architecture and training recipe as the target.
2. Query them with their own training data (members) and held-out data (non-members) and record the softmax outputs.
3. Train one binary attack classifier (logistic regression) per CIFAR-10 class on these outputs. Features are the softmax vector plus the one-hot true label, standardized; members and non-members are balanced per class.
4. Query the FL target model and let the attack models decide member or non-member.
5. Evaluate with ROC-AUC (primary metric), plus accuracy, precision, recall and F1.

---

## Getting started

### 1. Install

Requires Python 3.10–3.12 (developed with 3.12) and [Poetry](https://python-poetry.org/).
<!-- TODO: verify the Python range against pyproject.toml -->

```bash
git clone https://github.com/JonasGreim/Federated-Learning-MIA-Analysis.git
cd Federated-Learning-MIA-Analysis

poetry env use python3.12
poetry install

wandb login   # metrics are logged to Weights & Biases and also stored locally as JSON
```

### 2. Try the attack on an example model (fastest start)

A pre-trained FL checkpoint (10 rounds, 10 local epochs, IID data) is included in `model_checkpoints_target/example`, so you can run the attack without training anything:

```bash
python3 run_mia_experiment.py
```

Run a specific attack config:

```bash
python3 run_mia_experiment.py mia=run1
```

### 3. Full pipeline: train a federated model, then attack it

```bash
# 1) Train the FL target model (checkpoints are saved every 5th round)
python3 run_flower_experiment.py flower=run1

# 2) Attack the resulting checkpoint
python3 run_mia_experiment.py mia=run1
```

The default device is GPU. To run on CPU, set `device: cpu` in
[`experiments_conf/flower/base.yaml`](experiments_conf/flower/base.yaml) and
[`experiments_conf/mia/base.yaml`](experiments_conf/mia/base.yaml).
The Flower config also defines the client resources.

---

## Project structure

```text
.
├── flower/                       # FL system: models, custom FedAvg strategy, data splitting
├── membership_inference_attack/  # Shadow-model MIA implementation
├── experiments_conf/             # Hydra configs (Flower + MIA experiments)
├── experiments_conf_types/       # Dataclasses that type the configs
├── experiments_creation_scritps/ # Scripts that generate experiment configs
├── experiments_data_analysis/    # Aggregated data, figures, per-run metrics
├── hpc_slurm_scripts/            # SLURM batch scripts for cluster runs
├── model_checkpoints_target/     # Saved FL target models (example included)
├── run_flower_experiment.py      # Train a federated target model
├── run_mia_experiment.py         # Run the membership inference attack
├── run_multiple_experiments.py   # Run experiment grids
└── path_settings.py              # Output folders for checkpoints, logs, metrics
```

Analysis scripts in `experiments_data_analysis/` aggregate the per-run metrics into the tables and figures of the thesis (set your W&B entity and project in `file_name_settings.py`).

## Extending the project

- **Add a model:** put it in [`flower/models`](flower/models) and register it in [`flower/utils/model_factory.py`](flower/utils/model_factory.py), then use it in the configs.
- **Change the aggregation:** see [`flower/strategies/custom_weighted_fedavg.py`](flower/strategies/custom_weighted_fedavg.py). It extends Flower's FedAvg with model checkpointing, W&B logging and JSON metrics.
- **Change split sizes or dataset:** adjust the constants in [`split_cifar10_mia.py`](flower/utils/split_cifar10_mia.py) and delete the `dataset_splits` folder so the splits are regenerated. For another dataset, also update the model input shape and the class names in the MIA config.
- **Run on a cluster:** see [`hpc_slurm_scripts/`](hpc_slurm_scripts). Use `path_settings.py` to give each experiment its own checkpoint, log and metrics folders.

---

## Scope and limitations

- Only the **final global model** is attacked, by a **passive black-box** attacker. White-box attacks, active attackers and attacks on individual client models are out of scope.
- One dataset (CIFAR-10), two architectures, FedAvg, and 5 clients in the main experiment.
- The shadow models are trained centrally, and the attacker knows the target architecture and training procedure.
- The analysis is **correlational**. It shows associations, not causal explanations.
- No defenses such as differential privacy were evaluated.

**Possible next steps** (from the thesis): train the shadow models in a federated way, test stronger and active attackers, add client sampling and other aggregation methods (FedProx, SCAFFOLD), vary model capacity more systematically, and evaluate defenses like differential privacy.

## References

- R. Shokri, M. Stronati, C. Song, V. Shmatikov. *Membership Inference Attacks Against Machine Learning Models.* IEEE S&P, 2017.
- H. B. McMahan et al. *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS, 2017 (FedAvg).
- S. Yeom et al. *Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting.* IEEE CSF, 2018.
- A. Salem et al. *ML-Leaks: Model and Data Independent Membership Inference Attacks and Defenses on Machine Learning Models.* NDSS, 2019.
- D. J. Beutel et al. *Flower: A Friendly Federated Learning Research Framework.* 2020.