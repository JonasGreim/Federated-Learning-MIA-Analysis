# Membership Inference Attacks on Federated Learning

**How do model architecture and training choices affect membership leakage from a federated global model?**

This repository contains the experimental framework developed for my Master's thesis at Leipzig University. It combines **Federated Learning (FL)** with a black-box **Membership Inference Attack (MIA)** to investigate how model architecture, client data heterogeneity, local training, and regularization affect the privacy leakage of the final global model.

The project is built with **PyTorch**, **Flower**, **Hydra**, **scikit-learn**, and **Weights & Biases**. Large experiment grids were executed on the PAULA HPC cluster using **SLURM**.

> **Master's thesis, Computer Science — Data Privacy and Security, Leipzig University, 2026**
>
> *Einfluss von Trainingsparametern und Modellarchitekturen auf die Verwundbarkeit von Federated Learning gegenüber Membership-Inference-Angriffen*
>
> <!-- TODO: add thesis PDF, e.g. [Read the thesis (PDF, German)](docs/thesis.pdf) -->

---

## Key findings

The main study evaluates **36 federated learning configurations**:

- 2 model architectures
- 3 client data distributions
- 3 local-epoch settings
- with and without regularization

Each final global model is evaluated using the same black-box Membership Inference Attack.

- **The attack achieved above-random performance in all 36 configurations**, with ROC-AUC values between **0.57 and 0.81**.
- **Model architecture was the strongest observed factor.** ResNet-18 reached AUC values of **0.74–0.81** (median ≈ **0.78**), compared with **0.57–0.67** (median ≈ **0.63**) for the smaller Shokri-CNN.
- **Training parameters had smaller and strongly architecture-dependent effects.** Data heterogeneity, local epochs, and regularization did not produce a single monotonic relationship with attack strength.
- **Overfitting was related to attack strength, but did not fully explain it.** The Overfitting-Gap-Loss showed a positive overall correlation with MIA-AUC, while configurations with comparable overfitting could still exhibit noticeably different attack strengths.

<!--
Recommended: export Figure 5.2 from the thesis and add it here.

Example:
![MIA-AUC by model architecture](images/mia_auc_by_architecture.png)
-->

### Example: heterogeneity increases overfitting, but not attack strength

The following configurations use no regularization and one local epoch per communication round:

| Model | Client data distribution | Overfitting Gap (OGL) | Attack AUC |
| --- | --- | ---: | ---: |
| Shokri-CNN | IID | 1.01 | 0.66 |
| Shokri-CNN | semi-non-IID (α = 0.5) | 1.19 | 0.63 |
| Shokri-CNN | non-IID (α = 0.1) | 1.40 | 0.65 |
| ResNet-18 | IID | 2.12 | 0.81 |
| ResNet-18 | semi-non-IID (α = 0.5) | 2.45 | 0.80 |
| ResNet-18 | non-IID (α = 0.1) | 3.09 | 0.79 |

Increasing client-side data heterogeneity clearly increases the measured overfitting gap in this example, while attack AUC remains comparatively stable or decreases slightly.

This illustrates why overfitting is useful as a **risk indicator**, but not as a direct measure of Membership Inference vulnerability.

---

## Quick start

### Requirements

The project supports:

```text
Python >= 3.10 and < 3.13

The examples below use Python 3.12.
1. Install
git clone https://github.com/JonasGreim/Federated-Learning-MIA-Analysis.git
cd Federated-Learning-MIA-Analysis

poetry env use python3.12
poetry install

Experiments are logged to Weights & Biases and also stored locally as JSON:
wandb login

2. Run the attack on the included example model
The fastest way to try the project is to run the MIA directly.
A pre-trained federated target-model checkpoint is included in:
model_checkpoints_target/example/

Run the default attack:
python3 run_mia_experiment.py

Or select a specific Hydra configuration:
python3 run_mia_experiment.py mia=run1

3. Run the full pipeline
First train a federated target model:
python3 run_flower_experiment.py flower=run1

The global model is checkpointed during training and stored under:
model_checkpoints_target/

Then run the Membership Inference Attack using a MIA configuration that points to the desired checkpoint:
python3 run_mia_experiment.py mia=run1

The default configuration uses CUDA. To run on CPU, set:
device: "cpu"

in both:
experiments_conf/flower/base.yaml
experiments_conf/mia/base.yaml

How the experiment works
#chatgpt-mermaid-_r_3p0_{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI","Helvetica","Apple Color Emoji","Arial",sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;fill:rgb(237, 237, 237);}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#chatgpt-mermaid-_r_3p0_ .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_3p0_ .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_3p0_ .error-icon{fill:rgb(27, 27, 27);}#chatgpt-mermaid-_r_3p0_ .error-text{fill:rgb(237, 237, 237);stroke:rgb(237, 237, 237);}#chatgpt-mermaid-_r_3p0_ .edge-thickness-normal{stroke-width:1px;}#chatgpt-mermaid-_r_3p0_ .edge-thickness-thick{stroke-width:3.5px;}#chatgpt-mermaid-_r_3p0_ .edge-pattern-solid{stroke-dasharray:0;}#chatgpt-mermaid-_r_3p0_ .edge-thickness-invisible{stroke-width:0;fill:none;}#chatgpt-mermaid-_r_3p0_ .edge-pattern-dashed{stroke-dasharray:3;}#chatgpt-mermaid-_r_3p0_ .edge-pattern-dotted{stroke-dasharray:2;}#chatgpt-mermaid-_r_3p0_ .marker{fill:rgb(175, 175, 175);stroke:rgb(175, 175, 175);}#chatgpt-mermaid-_r_3p0_ .marker.cross{stroke:rgb(175, 175, 175);}#chatgpt-mermaid-_r_3p0_ svg{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI","Helvetica","Apple Color Emoji","Arial",sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;}#chatgpt-mermaid-_r_3p0_ p{margin:0;}#chatgpt-mermaid-_r_3p0_ .label{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI","Helvetica","Apple Color Emoji","Arial",sans-serif,"Segoe UI Emoji","Segoe UI Symbol";color:rgb(237, 237, 237);}#chatgpt-mermaid-_r_3p0_ .cluster-label text{fill:rgb(237, 237, 237);}#chatgpt-mermaid-_r_3p0_ .cluster-label span{color:rgb(237, 237, 237);}#chatgpt-mermaid-_r_3p0_ .cluster-label span p{background-color:transparent;}#chatgpt-mermaid-_r_3p0_ .label text,#chatgpt-mermaid-_r_3p0_ span{fill:rgb(237, 237, 237);color:rgb(237, 237, 237);}#chatgpt-mermaid-_r_3p0_ .node rect,#chatgpt-mermaid-_r_3p0_ .node circle,#chatgpt-mermaid-_r_3p0_ .node ellipse,#chatgpt-mermaid-_r_3p0_ .node polygon,#chatgpt-mermaid-_r_3p0_ .node path{fill:rgb(9, 23, 44);stroke:rgb(31, 78, 148);stroke-width:1px;}#chatgpt-mermaid-_r_3p0_ .rough-node .label text,#chatgpt-mermaid-_r_3p0_ .node .label text,#chatgpt-mermaid-_r_3p0_ .image-shape .label,#chatgpt-mermaid-_r_3p0_ .icon-shape .label{text-anchor:middle;}#chatgpt-mermaid-_r_3p0_ .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#chatgpt-mermaid-_r_3p0_ .rough-node .label,#chatgpt-mermaid-_r_3p0_ .node .label,#chatgpt-mermaid-_r_3p0_ .image-shape .label,#chatgpt-mermaid-_r_3p0_ .icon-shape .label{text-align:center;}#chatgpt-mermaid-_r_3p0_ .node.clickable{cursor:pointer;}#chatgpt-mermaid-_r_3p0_ .root .anchor path{fill:rgb(175, 175, 175)!important;stroke-width:0;stroke:rgb(175, 175, 175);}#chatgpt-mermaid-_r_3p0_ .arrowheadPath{fill:rgb(175, 175, 175);}#chatgpt-mermaid-_r_3p0_ .edgePath .path{stroke:rgb(175, 175, 175);stroke-width:1px;}#chatgpt-mermaid-_r_3p0_ .flowchart-link{stroke:rgb(175, 175, 175);fill:none;}#chatgpt-mermaid-_r_3p0_ .edgeLabel{background-color:rgb(0, 0, 0);text-align:center;}#chatgpt-mermaid-_r_3p0_ .edgeLabel p{background-color:rgb(0, 0, 0);}#chatgpt-mermaid-_r_3p0_ .edgeLabel rect{opacity:0.5;background-color:rgb(0, 0, 0);fill:rgb(0, 0, 0);}#chatgpt-mermaid-_r_3p0_ .labelBkg{background-color:rgba(0, 0, 0, 0.5);}#chatgpt-mermaid-_r_3p0_ .cluster rect{fill:rgb(27, 27, 27);stroke:rgba(255, 255, 255, 0.15);stroke-width:1px;}#chatgpt-mermaid-_r_3p0_ .cluster text{fill:rgb(237, 237, 237);}#chatgpt-mermaid-_r_3p0_ .cluster span{color:rgb(237, 237, 237);}#chatgpt-mermaid-_r_3p0_ div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI","Helvetica","Apple Color Emoji","Arial",sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:12px;background:rgb(27, 27, 27);border:1px solid rgba(255, 255, 255, 0.15);border-radius:2px;pointer-events:none;z-index:100;}#chatgpt-mermaid-_r_3p0_ .flowchartTitleText{text-anchor:middle;font-size:18px;fill:rgb(237, 237, 237);}#chatgpt-mermaid-_r_3p0_ rect.text{fill:none;stroke-width:0;}#chatgpt-mermaid-_r_3p0_ .icon-shape,#chatgpt-mermaid-_r_3p0_ .image-shape{background-color:rgb(0, 0, 0);text-align:center;}#chatgpt-mermaid-_r_3p0_ .icon-shape p,#chatgpt-mermaid-_r_3p0_ .image-shape p{background-color:rgb(0, 0, 0);padding:2px;}#chatgpt-mermaid-_r_3p0_ .icon-shape .label rect,#chatgpt-mermaid-_r_3p0_ .image-shape .label rect{opacity:0.5;background-color:rgb(0, 0, 0);fill:rgb(0, 0, 0);}#chatgpt-mermaid-_r_3p0_ .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#chatgpt-mermaid-_r_3p0_ .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#chatgpt-mermaid-_r_3p0_ .node .neo-node{stroke:rgb(31, 78, 148);}#chatgpt-mermaid-_r_3p0_ [data-look="neo"].node rect,#chatgpt-mermaid-_r_3p0_ [data-look="neo"].cluster rect,#chatgpt-mermaid-_r_3p0_ [data-look="neo"].node polygon{stroke:url(#chatgpt-mermaid-_r_3p0_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_3p0_ [data-look="neo"].swimlane.cluster rect{filter:none;}#chatgpt-mermaid-_r_3p0_ [data-look="neo"].node path{stroke:url(#chatgpt-mermaid-_r_3p0_-gradient);stroke-width:1px;}#chatgpt-mermaid-_r_3p0_ [data-look="neo"].node .outer-path{filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_3p0_ [data-look="neo"].node .neo-line path{stroke:rgb(31, 78, 148);filter:none;}#chatgpt-mermaid-_r_3p0_ [data-look="neo"].node circle{stroke:url(#chatgpt-mermaid-_r_3p0_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_3p0_ [data-look="neo"].node circle .state-start{fill:#000000;}#chatgpt-mermaid-_r_3p0_ [data-look="neo"].icon-shape .icon{fill:url(#chatgpt-mermaid-_r_3p0_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_3p0_ [data-look="neo"].icon-shape .icon-neo path{stroke:url(#chatgpt-mermaid-_r_3p0_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_3p0_ .node text{font-size:14px;font-weight:600;letter-spacing:normal;fill:rgb(153, 206, 255);}#chatgpt-mermaid-_r_3p0_ .edgeLabels text{font-size:13px;font-weight:600;letter-spacing:-0.08px;fill:rgb(153, 206, 255);}#chatgpt-mermaid-_r_3p0_ .node tspan[font-weight="normal"],#chatgpt-mermaid-_r_3p0_ .edgeLabels tspan[font-weight="normal"]{font-weight:600;}#chatgpt-mermaid-_r_3p0_ .edgeLabel .label rect{opacity:1;rx:13px;ry:13px;fill:rgb(0, 14, 26);stroke:rgb(26, 62, 95);stroke-width:1px;}#chatgpt-mermaid-_r_3p0_ .node rect,#chatgpt-mermaid-_r_3p0_ .node circle,#chatgpt-mermaid-_r_3p0_ .node ellipse,#chatgpt-mermaid-_r_3p0_ .node polygon,#chatgpt-mermaid-_r_3p0_ .node path{fill:rgb(0, 40, 77);stroke:rgba(255, 255, 255, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_3p0_ .node rect{rx:16px;ry:16px;}#chatgpt-mermaid-_r_3p0_ .node.mermaid-decision .label-container{fill:rgb(0, 14, 26);stroke:rgb(26, 62, 95);stroke-dasharray:2px,2px;}#chatgpt-mermaid-_r_3p0_ .edgePaths .flowchart-link{stroke:rgb(175, 175, 175);stroke-width:1px;stroke-linecap:round;stroke-linejoin:round;}#chatgpt-mermaid-_r_3p0_ .marker{fill:rgb(175, 175, 175);stroke:rgb(175, 175, 175);}#chatgpt-mermaid-_r_3p0_ :root{--mermaid-font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI","Helvetica","Apple Color Emoji","Arial",sans-serif,"Segoe UI Emoji","Segoe UI Symbol";}CIFAR-10Disjoint dataset splitsFederated target-modeltrainingFinal global modelShadow-model dataShadow modelMember / non-memberfeaturesClass-specific attack modelsTarget-model predictionsMembership inferenceROC-AUC / Accuracy / F1




The experiment separates target-model training, attack-model training, and attack evaluation using strictly disjoint data partitions.
Membership Inference Attack
The attack is based on the shadow-model approach introduced by Shokri et al.
The goal is to infer whether a particular sample was used to train the final global federated model.
Threat model
The attacker is assumed to be a passive global black-box attacker.
The attacker can query the final global model and observe its prediction probabilities.
The attacker does not have access to:
- model parameters
- gradients
- local client models
- client training data
- intermediate client updates
- the federated training process
The attacker is assumed to know the target model architecture and training procedure and has access to separate data drawn from the same underlying distribution.
Attack pipeline
1. Train one or more shadow models on data that is strictly disjoint from the target model's training data.
2. Query each shadow model using:
   - samples used for shadow-model training (member)
   - samples not used for shadow-model training (non-member)
3. Extract the full Softmax probability vector and the one-hot encoded true class label.
4. Group the resulting attack data by class.
5. Balance member and non-member samples and standardize the features.
6. Train one binary logistic regression attack classifier per CIFAR-10 class.
7. Query the final federated target model using known member and non-member samples.
8. Use the corresponding class-specific attack model to predict membership.
The primary metric used throughout the thesis is ROC-AUC. Accuracy, Precision, Recall, and F1 are also recorded.
Experimental setup
Federated learning
The main experiment uses a synchronous horizontal FL system implemented with Flower.
- Aggregation: Federated Averaging (FedAvg)
- Clients: 5
- Client participation: all clients participate in every communication round
- Dataset: CIFAR-10
- Models: Shokri-CNN and ResNet-18
- Local epochs: 1, 5, or 10
- Client distributions: IID, semi-non-IID (α = 0.5), non-IID (α = 0.1)
- Regularization: no regularization or Weight Decay (5 × 10⁻⁴) + Dropout (0.5)
To isolate the effect of local training intensity, every client performs 100 local training epochs in total.
The number of communication rounds is therefore adjusted accordingly:
Local epochs per round	Communication rounds	Total local epochs
1	100	100
5	20	100
10	10	100


This keeps the total local training effort constant while changing how much local optimization happens between two global aggregations.
Data partitioning
CIFAR-10 is divided into four strictly disjoint partitions:
Split	Size	Purpose
D1	20,000	Federated target-model training
D2	10,000	Global target-model evaluation
D3	15,000	Shadow-model training
D4	15,000	Shadow-model testing


For the final MIA evaluation:
- D1 provides known member samples.
- D2 provides known non-member samples.
- D3 and D4 are used exclusively to construct the shadow-model attack data.
This separation prevents the attack model from being trained on data used to train the federated target model.
Overfitting
The primary overfitting indicator is the Overfitting-Gap-Loss (OGL).
It compares the aggregated client-side training loss with the server-side test loss of the global model.
The thesis uses OGL as an empirical proxy for generalization behavior and investigates whether it can also serve as an indicator of Membership Inference risk.
Validation experiments
Two additional experiments test whether important design choices substantially change the main findings.
Number of federated clients
The number of participating clients was varied between 2, 5, and 10.
Model	2 clients	5 clients	10 clients
Shokri-CNN	0.71	0.66	0.55
ResNet-18	0.81	0.82	0.81


For the Shokri-CNN, increasing the number of clients strongly reduced attack strength in this configuration.
For ResNet-18, attack performance remained almost unchanged.
This is another example of the strong architecture dependence observed throughout the experiments.
Number of shadow models
The attack was evaluated using 1, 3, 10, and 20 shadow models.
Increasing the number of shadow models did not systematically improve attack performance in the evaluated setup. Attack AUC was already stable with a single shadow model.
This result is limited to the considered threat model, where the attacker knows the target architecture and uses matching shadow models.
Project structure
.
├── flower/
│   ├── models/                     # Target and shadow model architectures
│   ├── strategies/                 # Custom Flower / FedAvg strategy
│   └── utils/                      # Data loading, logging, reproducibility, ...
│
├── membership_inference_attack/
│   └── mia_shokri.py               # Shadow-model MIA implementation
│
├── experiments_conf/
│   ├── flower/                     # Hydra configs for federated training
│   └── mia/                        # Hydra configs for attacks
│
├── experiments_conf_types/         # Typed configuration dataclasses
├── experiments_creation_scritps/  # Scripts for generating experiment configs
├── experiments_data_analysis/     # Aggregation, analysis, plots and tables
├── hpc_slurm_scripts/             # SLURM scripts for HPC execution
├── model_checkpoints_target/      # Saved global target models
├── images/                        # Images used by the project / README
│
├── run_flower_experiment.py       # Run federated target-model training
├── run_mia_experiment.py          # Run a Membership Inference Attack
├── run_multiple_experiments.py    # Execute experiment grids
├── run_flower_hpc.py              # HPC Flower execution
├── path_settings.py               # Output paths
└── pyproject.toml                 # Dependencies and Flower configuration

Each experiment stores metrics locally as JSON and logs them to Weights & Biases.
The scripts in experiments_data_analysis/ aggregate the experimental results and generate the tables and figures used in the thesis.
Configuration
Experiment configurations are managed with Hydra.
Federated-learning configurations are stored in:
experiments_conf/flower/

Membership-Inference configurations are stored in:
experiments_conf/mia/

The corresponding typed configuration definitions are located in:
experiments_conf_types/

Important FL parameters include:
num_server_rounds: 100
local_epochs: 1
weight_decay: 0.0
model: "simple_model"
iid_data_distribution: true
dirichlet_alpha: 0.0
num_clients: 5

Important MIA parameters include the number of shadow models, shadow-model architecture, number of shadow-training epochs, target-model checkpoint, and attack-data sizes.
Extending the project
Add another model
Add the architecture to:
flower/models/

and register it in the model factory:
flower/utils/model_factory.py

The model can then be referenced from the experiment configurations.
Change the aggregation strategy
The custom strategy is implemented in:
flower/strategies/custom_weighted_fedavg.py

It extends Flower's standard FedAvg workflow with project-specific functionality such as checkpointing and experiment logging.
Use another dataset
The current data pipeline is designed around CIFAR-10.
Using another dataset requires adapting the dataset loading and model input dimensions and updating the corresponding class information used by the attack.
Run experiment grids on HPC
SLURM scripts are available in:
hpc_slurm_scripts/

The large-scale experiments for the thesis were executed on the PAULA HPC cluster at Leipzig University.
Tech stack
Machine Learning: PyTorch, torchvision, scikit-learn
Federated Learning: Flower
Experiment configuration: Hydra, OmegaConf
Tracking & analysis: Weights & Biases, pandas, NumPy, matplotlib, seaborn
Dataset: CIFAR-10 / Hugging Face Datasets
Infrastructure: SLURM, GPU-based HPC execution
Scope and limitations
The results should be interpreted within the experimental scope of the thesis:
- The attack targets only the final aggregated global model.
- The threat model is a passive black-box attacker.
- White-box attacks, active attacks, and attacks against individual client models are not evaluated.
- The main study uses one dataset, two model architectures, and five clients.
- The analysis identifies empirical relationships and does not establish causal effects.
- The main experiment uses one shadow model; the additional shadow-model experiment only validates this choice for the considered setting.
- No formal privacy guarantees or dedicated privacy-preserving defense mechanisms are evaluated.
Master's thesis
This repository contains the experimental implementation developed for:
Einfluss von Trainingsparametern und Modellarchitekturen auf die Verwundbarkeit von Federated Learning gegenüber Membership-Inference-Angriffen
Master's thesis in Computer Science
Leipzig University
Institute of Computer Science
Data Privacy and Security
2026
The thesis investigates how model architecture and federated training parameters influence the vulnerability of the final global model to Membership Inference Attacks and how this vulnerability relates to model overfitting.
References
1. R. Shokri, M. Stronati, C. Song, V. Shmatikov.
   Membership Inference Attacks Against Machine Learning Models.
   IEEE Symposium on Security and Privacy, 2017.
2. H. B. McMahan et al.
   Communication-Efficient Learning of Deep Networks from Decentralized Data.
   AISTATS, 2017.
3. S. Yeom et al.
   Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting.
   IEEE Computer Security Foundations Symposium, 2018.
4. A. Salem et al.
   ML-Leaks: Model and Data Independent Membership Inference Attacks and Defenses on Machine Learning Models.
   NDSS, 2019.