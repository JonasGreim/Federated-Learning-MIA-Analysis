# my-awesome-app: A Flower / PyTorch app

## Install dependencies and project

```bash
pip install -e .
```

## Run with the Simulation Engine

In the `my-awesome-app` directory, use `flwr run` to run a local simulation:

```bash
flwr run .
```

Refer to the [How to Run Simulations](https://flower.ai/docs/framework/how-to-run-simulations.html) guide in the documentation for advice on how to optimize your simulations.

## Run with the Deployment Engine

> \[!NOTE\]
> An update to this example will show how to run this Flower application with the Deployment Engine and TLS certificates, or with Docker.

## Resources

- Flower website: [flower.ai](https://flower.ai/)
- Check the documentation: [flower.ai/docs](https://flower.ai/docs/)
- Give Flower a ⭐️ on GitHub: [GitHub](https://github.com/adap/flower)
- Join the Flower community!
  - [Flower Slack](https://flower.ai/join-slack/)
  - [Flower Discuss](https://discuss.flower.ai/)

setup:
- git clone
- use poetry as package manager with python 3.10
  - poetry install
- run simulation (with 10 client nodes)
  - flwr run .
- first run -> wandb login: API-key

dataset: https://huggingface.co/datasets/uoft-cs/cifar10
- 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- trainset of 50k images (80% train, 20% test) (5k images from each class)
- testset of 10k images 
- 32x32 pixels, 3 channels(rgb)
- size: 144MB

cluster:
- 10 clients (super nodes)
- num-server-rounds = 5 (get slitted on all cpu cores)
- fraction-fit = 0.5
- local-epochs = 1
- explain:
  - serverApp: client selection, client configuration, result aggregation (short lived process)
  - SuperLink: forwards task instructions to clients (SuperNodes) and receives task results back 
  - SuperNode: hold data, asks for tasks, executes tasks(training), and sends results back to the server 
  - ClientApp: local model training and evaluation, pre- and post-processing (short lived process)
  - (network communication is taken care of by Flower: SuperLink, SuperNode)

![architectures.png](images/architectures.png)

change dataset: (in task)
- choose huggingface dataset: dataset="uoft-cs/cifar10",
- check the column names in hugging face: batch["img"] or batch["image"]
- check if dataset is greyscale or rgb: -> change Net and Compose(ToTensor(), Normalize((0.5 or 0.5,0.5,0.5), ...)
- check size of dataset -> change Net

Changes:
 - dataset: "uoft-cs/cifar10"
 - partitioner: non-iid (DirichletPartitioner with alpha 0.5)

callbacks: (in Strategy FedAvg, serverApp)
- how to aggregate metrics sent back from the clients app into strategy (weighted_average)
  (it is also possible to evaluate the model globally/centralized on the server app if there is a global evaluation dataset)
- learning decreases with higher round number (perform fit method in a different way)
- centralized evaluation on the server app after each global model aggregation round 

costume strategy: (global model aggregation)
- add pytorch model checkpoints for each round 
- push metrics to wandb (weights and biases) for each round 
- create json file to store metrics


what i could do better but won t:
- add more attack Attack Model Diversity for  more black box(CatBoost, XGBoost, Logistic Regression)
- Use logits, loss values, or gradient norms (if white-box is permitted)
- Implement top-k probability truncation to simulate black-box API constraints.


Helps against overfitting -> no testing:
- Model Calibration: temperature scaling to flatten softmax probabilities

Run:
with GPU: (change in toml device parameter)
```bash
flwr run . local-simulation-gpu
```
with CPU: (change in toml device parameter)
```bash
flwr run . 
```

Difference to Shokri:
 - Attack Model Type RandomForestClassifier instead of small MLP
 - number of Shadow Models: 4+ instead of 1
 - extra features
 - Balanced Member/Non-Member Training (not done in Shokri)
 - shadow model datasets are the same size as the target model’s training set
    - currently only if 1 shadow model is used
    - testset should also be the same size as the target model’s training set (splitted)
 -  code currently implements one attack model per class across all shadow models, not per shadow model per class
  improved versions of MIA from later papers (like Salem et al. 2018 or Yeom et al. 2018) where fewer shadow models are used or models are shared.

