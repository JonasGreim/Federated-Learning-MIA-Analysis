# Thesis facts (verified against the final thesis)

Source: *Einfluss von Trainingsparametern und Modellarchitekturen auf die Verwundbarkeit von Federated Learning gegenüber Membership-Inference-Angriffen* (Jonas Greim, 2026, 86 pages). Page numbers are the printed page numbers. Use this file as the source of truth for numbers and wording. Add new facts with a page number.

## Study design (pp. 13-17, 24, 29-31)

- Horizontal, synchronous FL, FedAvg, 5 clients, all clients participate every round (main experiment). Framework: Flower.
- Dataset: CIFAR-10. Splits: D1 = 20,000 (FL training, members), D2 = 10,000 (from original test set, non-members / server evaluation), D3 = 15,000 (shadow training), D4 = 15,000 (shadow test). Strictly disjoint.
- Each client partition is split 80/20 into local train/test (p. 25).
- Architectures: Shokri-CNN (2 conv + pooling layers, 1 FC layer) and ResNet-18.
- Parameters: local epochs per round 1 / 5 / 10 (rounds adjusted so every client has 100 local epochs in total: 100 / 20 / 10 rounds); data distribution IID, semi-non-IID (Dirichlet alpha = 0.5), non-IID (alpha = 0.1); regularization none or weight decay 5e-4 + dropout 0.5.
- Full grid: 3 x 3 x 2 x 2 = 36 target-model configurations, one attack each.
- Shadow models: trained centrally, same architecture/optimizer/regularization as clients, 100 epochs. Main experiment uses ONE shadow model (D3 15,000 / D4 15,000).
- Attack (Shokri et al. style, p. 29-31): passive black-box attacker, softmax outputs only. Features = full softmax vector + one-hot true label, standardized. One binary **logistic regression** attack model **per class**, members/non-members balanced per class by resampling. Evaluation: D1 members vs. D2 non-members on the checkpoint of the final global model.
- Metrics: accuracy, precision, recall, F1, ROC-AUC (MIA-AUC = primary). Fixed seeds; metrics stored persistently.
- Overfitting indicator: Overfitting-Gap-Loss (OGL) = aggregated client-side training loss vs. server-side test loss of the global model.
- Infrastructure (p. 26): the thesis names the GPU cluster **PAULA**; nodes have 4x NVIDIA Tesla A30 (24 GB) and Rocky Linux 9; FL training uses 6 nodes (1 server + 5 clients), while the MIA runs on a single node.
- Out of scope (p. 2-3, 33): white-box attacks, active attacks, attacks on client models, formal privacy guarantees, defenses such as differential privacy.

## Main results

### Overall (pp. 42-46)

- Attack AUC is above 0.5 in all 36 configurations; range 0.57 to 0.81.
- Shokri-CNN: AUC 0.57-0.67, median about 0.63. ResNet-18: AUC 0.74-0.81, median about 0.78 (computed 0.775).
- Regularization, all configs: median about 0.70 without vs. about 0.69 with, practically negligible, ranges overlap (p. 45-46).
- Regularization per architecture (computed from Table 5.1): Shokri-CNN median 0.64 -> 0.60; ResNet-18 median 0.79 -> 0.77.
- Regularization raises OGL for the Shokri-CNN in most configurations (e.g. IID, 1 epoch: 1.01 -> 1.18) and lowers it for ResNet-18 (e.g. IID, 1 epoch: 2.12 -> 1.98).
- Data distribution, local epochs, regularization: moderate, often non-monotonic effects that depend on architecture (Sections 5.2.4-5.2.7, 6.2).
- Overfitting vs. attack strength: positive but not deterministic relationship (Pearson correlation over all configs); much weaker within a fixed architecture; similar OGL can give clearly different AUC (Sections 5.3, 6.1). Analysis is correlational, not causal (p. 50).
- ResNet-18 has higher AUC than Shokri-CNN even at comparable OGL (Section 6.1).
- Heterogeneity example (computed from Table 5.1, 1 local epoch, no regularization): OGL rises with heterogeneity (Shokri-CNN 1.01 / 1.19 / 1.40, ResNet-18 2.12 / 2.45 / 3.09) while AUC varies by at most 0.03 (Shokri-CNN 0.66 / 0.63 / 0.65, ResNet-18 0.81 / 0.80 / 0.79). This does not hold everywhere: ResNet-18 with 10 local epochs and no regularization has OGL 2.02 / 1.97 / 1.94 and AUC 0.79 / 0.74 / 0.74.

### Validation experiments (pp. 57-60, Appendix D)

- Number of clients (Table A.2): Shokri-CNN AUC 0.71 / 0.66 / 0.55 and OGL 1.50 / 1.01 / 0.21 for 2 / 5 / 10 clients. ResNet-18 AUC 0.81 / 0.82 / 0.81 and OGL 1.84 / 2.13 / 2.12. (Setup: 1 local epoch, no regularization; the 5-client reference values differ by 0.01 from Table 5.1 for ResNet-18.)
- Number of shadow models (1, 3, 10, 20): attack strength stabilizes early, no systematic increase. With one shadow model D3 (15,000) is used; with several, random subsets of 10,000 from D3. Valid for an attacker who knows architecture and training procedure (p. 59-60).

## Table 5.1 (pp. 37-38): all 36 configurations

| Model | Regularization | Data distribution | Local epochs | OGL | MIA-AUC |
|---|---|---|---:|---:|---:|
| Shokri-CNN | no | IID | 1 | 1.01 | 0.66 |
| Shokri-CNN | no | IID | 5 | 1.30 | 0.67 |
| Shokri-CNN | no | IID | 10 | 1.31 | 0.64 |
| Shokri-CNN | no | semi-non-IID | 1 | 1.19 | 0.63 |
| Shokri-CNN | no | semi-non-IID | 5 | 1.35 | 0.62 |
| Shokri-CNN | no | semi-non-IID | 10 | 1.35 | 0.60 |
| Shokri-CNN | no | non-IID | 1 | 1.40 | 0.65 |
| Shokri-CNN | no | non-IID | 5 | 1.50 | 0.64 |
| Shokri-CNN | no | non-IID | 10 | 1.54 | 0.63 |
| Shokri-CNN | yes | IID | 1 | 1.18 | 0.59 |
| Shokri-CNN | yes | IID | 5 | 1.37 | 0.64 |
| Shokri-CNN | yes | IID | 10 | 1.38 | 0.63 |
| Shokri-CNN | yes | semi-non-IID | 1 | 1.27 | 0.57 |
| Shokri-CNN | yes | semi-non-IID | 5 | 1.40 | 0.60 |
| Shokri-CNN | yes | semi-non-IID | 10 | 1.40 | 0.60 |
| Shokri-CNN | yes | non-IID | 1 | 1.34 | 0.58 |
| Shokri-CNN | yes | non-IID | 5 | 1.55 | 0.62 |
| Shokri-CNN | yes | non-IID | 10 | 1.65 | 0.63 |
| ResNet-18 | no | IID | 1 | 2.12 | 0.81 |
| ResNet-18 | no | IID | 5 | 2.10 | 0.81 |
| ResNet-18 | no | IID | 10 | 2.02 | 0.79 |
| ResNet-18 | no | semi-non-IID | 1 | 2.45 | 0.80 |
| ResNet-18 | no | semi-non-IID | 5 | 2.13 | 0.78 |
| ResNet-18 | no | semi-non-IID | 10 | 1.97 | 0.74 |
| ResNet-18 | no | non-IID | 1 | 3.09 | 0.79 |
| ResNet-18 | no | non-IID | 5 | 2.28 | 0.77 |
| ResNet-18 | no | non-IID | 10 | 1.94 | 0.74 |
| ResNet-18 | yes | IID | 1 | 1.98 | 0.78 |
| ResNet-18 | yes | IID | 5 | 1.83 | 0.80 |
| ResNet-18 | yes | IID | 10 | 1.86 | 0.79 |
| ResNet-18 | yes | semi-non-IID | 1 | 2.17 | 0.77 |
| ResNet-18 | yes | semi-non-IID | 5 | 1.78 | 0.77 |
| ResNet-18 | yes | semi-non-IID | 10 | 1.72 | 0.74 |
| ResNet-18 | yes | non-IID | 1 | 2.52 | 0.77 |
| ResNet-18 | yes | non-IID | 5 | 1.82 | 0.77 |
| ResNet-18 | yes | non-IID | 10 | 1.62 | 0.75 |

## Approved wording (use as is or close to it)

- "The attack achieved above-random performance in all 36 evaluated configurations (AUC 0.57-0.81)."
- "Model architecture was the strongest observed factor."
- "Regularization had architecture-dependent effects."
- "Overfitting was positively related to attack strength, but did not fully explain it."
- "A single shadow model was sufficient in the evaluated setup."

## Future work stated in the thesis (p. 67-68)

- Train shadow models in a federated way; test stronger and active attackers (e.g. manipulating local updates or aggregation).
- Client sampling / partial participation; other aggregation methods (FedProx, SCAFFOLD).
- Systematic variation of model capacity (depth, width).
- Dynamics of overfitting and attack signals over communication rounds (key epochs).
- Defenses such as differential privacy are named as a possible extension (p. 68).

## References used in the README (pp. 69-70)

- B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas. *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS, 2017.
- R. Shokri, M. Stronati, C. Song, and V. Shmatikov. *Membership Inference Attacks Against Machine Learning Models.* IEEE Symposium on Security and Privacy, 2017.
- S. Yeom, I. Giacomelli, M. Fredrikson, and S. Jha. *Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting.* IEEE Computer Security Foundations Symposium, 2018.
- A. Salem, Y. Zhang, M. Humbert, P. Berrang, M. Fritz, and M. Backes. *ML-Leaks: Model and Data Independent Membership Inference Attacks and Defenses on Machine Learning Models.* NDSS, 2019.

## Items to verify against the repo (not facts from the thesis)

- Python version range, license file, cluster name, config key names, default device and checkpoint frequency, whether W&B login is mandatory.
