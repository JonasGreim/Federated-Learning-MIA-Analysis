import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
import numpy as np
import os
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils import resample
from my_awesome_app.utils.types_config_mia import MiaConfig
from my_awesome_app.utils.task import get_transforms_custom, seed_everything, seed_worker, release_model, create_model
from collections import defaultdict
import re
import time
import hydra
from pathlib import Path
import wandb
from omegaconf import OmegaConf
from my_awesome_app.utils.wandb_logging import log_class_distribution, log_training_to_wandb, log_per_class_metrics, \
    log_overall_metrics_with_error_bars


# === Utility Functions ===
def load_data(config) -> tuple[Subset, Subset, Subset, Subset]:
    root_dir = Path(config.paths.current_root).parent
    data_dir = config.paths.data_dir
    split_dir = config.paths.split_dir
    data_dir_path = os.path.join(root_dir, data_dir)

    train_dataset = CIFAR10(root=data_dir_path, train=True, download=True, transform=get_transforms_custom())

    # load split indices
    D1 = np.load(os.path.join(root_dir, split_dir, "D1_indices.npy")).tolist()
    D2 = np.load(os.path.join(root_dir, split_dir, "D2_indices.npy")).tolist()
    D3 = np.load(os.path.join(root_dir, split_dir, "D3_indices.npy")).tolist()
    D4 = np.load(os.path.join(root_dir, split_dir, "D4_indices.npy")).tolist()
    # D1: train set for target model (10000 samples, 1000 per class)
    # D2: test set for target model (10000 samples, 1000 per class)
    # D3: train set for shadow model (15000 samples, 1500 per class)
    # D4: test set for shadow model (15000 samples, 1500 per class)

    target_train = Subset(train_dataset, D1)
    target_test = Subset(train_dataset, D2)
    shadow_train = Subset(train_dataset, D3)
    shadow_test = Subset(train_dataset, D4)

    log_class_distribution(shadow_train, wandb_cluster_name="shadow_train_distribution",
                           wandb_plot_prefix="shadow_pool_data")

    return shadow_train, target_train, target_test, shadow_test


def sample_shadow_datasets_with_overlap(shadow_train_dataset, shadow_test_dataset, config) -> tuple[
    list[Subset], list[Subset]]:
    num_shadow_models = config.parameters.num_shadow_models
    seed: int = config.parameters_static.seed
    train_size: int = config.parameters.train_size
    test_size: int = config.parameters.test_size

    all_indices = np.array(shadow_train_dataset.indices)
    test_indices_pool = np.array(shadow_test_dataset.indices)

    shadow_train_sets = []
    shadow_test_sets = []

    for i in range(num_shadow_models):
        rng = np.random.RandomState(seed + i)  # different seed for each shadow model

        # Sample train and test sets without replacement
        train_indices = rng.choice(all_indices, size=train_size, replace=False)
        test_indices = rng.choice(test_indices_pool, size=test_size, replace=False)
        train_subset = Subset(shadow_train_dataset.dataset, train_indices)

        log_class_distribution(train_subset, wandb_cluster_name="shadow_train_distribution",
                               wandb_plot_prefix=f"shadow_model_{i + 1}")

        shadow_train_sets.append(train_subset)
        shadow_test_sets.append(Subset(shadow_test_dataset.dataset, test_indices))

    return shadow_train_sets, shadow_test_sets


def train_model(model, dataloader, epochs, learning_rate, learning_rate_decay, device) -> tuple[nn.Module, list[dict]]:
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda e: 1 / (1 + learning_rate_decay * e)
    )
    criterion = nn.CrossEntropyLoss()
    model.train()

    history = []

    for epoch in range(epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "train_accuracy": epoch_accuracy,
        })

    return model, history


def extract_features_by_class(model, dataloader, label_indicator, num_classes, device) -> dict:
    model.eval()
    class_features = defaultdict(list)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)

            probs_np = probs.cpu().numpy()
            labels_np = labels.cpu().numpy()

            for prob_vec, cls_label in zip(probs_np, labels_np):
                class_one_hot = np.eye(num_classes)[cls_label]  # One-hot encoding of class label
                input_vector = np.concatenate([prob_vec, class_one_hot])
                class_features[cls_label].append((input_vector, label_indicator))

    return class_features


def train_all_shadow_models_and_collect_features(shadow_train_subsets, shadow_test_subsets, config, device) -> list[
    dict]:
    seed = config.parameters_static.seed
    num_classes = config.parameters.num_classes
    batch_size = config.parameters_static.batch_size
    shadow_epochs = config.parameters.shadow_epochs
    learning_rate = config.parameters_static.learning_rate
    learning_rate_decay = config.parameters_static.learning_rate_decay
    model_name = config.parameters.model_arch

    per_shadow_per_class_data = []

    for i, (train_subset, test_subset) in enumerate(zip(shadow_train_subsets, shadow_test_subsets)):
        print(f"🔄 Training Shadow Model {i + 1}/{len(shadow_train_subsets)}")
        model_arch = create_model(model_name)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                  worker_init_fn=seed_worker,
                                  generator=torch.Generator().manual_seed(seed + i), num_workers=2)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)

        # Measure training time
        start_time = time.time()
        model, history = train_model(model=model_arch, dataloader=train_loader, epochs=shadow_epochs,
                                     learning_rate=learning_rate, learning_rate_decay=learning_rate_decay,
                                     device=device)

        training_time = time.time() - start_time
        log_training_to_wandb(history, training_time, model_idx=i)

        print(f"⏱️ Shadow Model {i + 1} Training Time: {training_time:.2f} seconds")

        member_features = extract_features_by_class(model=model, dataloader=train_loader, label_indicator=1,
                                                    num_classes=num_classes, device=device)
        nonmember_features = extract_features_by_class(model=model, dataloader=test_loader, label_indicator=0,
                                                       num_classes=num_classes, device=device)

        per_class_data = defaultdict(lambda: {'member': [], 'nonmember': []})

        print(f"📊 Shadow Model {i + 1} — Per-Class Member/Non-Member Counts:")
        for cls in range(num_classes):
            num_members = len(member_features.get(cls, []))
            num_nonmembers = len(nonmember_features.get(cls, []))
            per_class_data[cls]['member'].extend(member_features.get(cls, []))
            per_class_data[cls]['nonmember'].extend(nonmember_features.get(cls, []))
            print(f"    Class {cls}: Members = {num_members}, Non-Members = {num_nonmembers}")

        per_shadow_per_class_data.append(per_class_data)
        release_model(model, device.type)

    return per_shadow_per_class_data


def train_per_class_attack_models(per_shadow_per_class_data, config) -> tuple[dict, dict]:
    """
    Trains one attack model per class using features pooled from all shadow models (faithful to Shokri et al.)
    """
    num_classes = config.parameters.num_classes
    seed = config.parameters_static.seed

    pooled_per_class_data = defaultdict(lambda: {'member': [], 'nonmember': []})

    # Pool features from all shadows
    for shadow_idx, class_data in enumerate(per_shadow_per_class_data):
        for cls in range(num_classes):
            pooled_per_class_data[cls]['member'].extend(class_data[cls]['member'])
            pooled_per_class_data[cls]['nonmember'].extend(class_data[cls]['nonmember'])

    attack_models = {}
    scalers = {}

    start_time_attack_train = time.time()

    # Train one attack model per class
    for cls in range(num_classes):
        member = pooled_per_class_data[cls]['member']
        nonmember = pooled_per_class_data[cls]['nonmember']

        if not member or not nonmember:
            print(f"⚠️ Skipping Class {cls}: insufficient data (Members: {len(member)}, Non-members: {len(nonmember)})")
            continue

        X_member, y_member = zip(*member)
        X_nonmember, y_nonmember = zip(*nonmember)

        # Balance members and non-members (Shokri likely used undersampling)
        n_samples = min(len(X_member), len(X_nonmember))

        X_member, y_member = resample(X_member, y_member, n_samples=n_samples, random_state=seed, replace=True)
        X_nonmember, y_nonmember = resample(X_nonmember, y_nonmember, n_samples=n_samples, random_state=seed,
                                            replace=True)

        X = np.vstack((X_member, X_nonmember))
        y = np.array(y_member + y_nonmember)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # In paper Shokri et al. used logistic regression, but RandomForest is better for this task
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X_scaled, y)

        # clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=SEED)
        # clf.fit(X_scaled, y)

        attack_models[cls] = clf
        scalers[cls] = scaler

        print(f"✅ Trained attack model for Class {cls} (Samples: {len(y)})")
        print(f" Class before balancing:  Members: {len(member)}, Non-members: {len(nonmember)}")

    total_attack_training_time = time.time() - start_time_attack_train
    print(f"⏱️ Total Attack Model Training Time: {total_attack_training_time:.2f} seconds")

    return attack_models, scalers


def evaluate_attack_models(attack_models, scalers, target_train, target_test, config, device) -> None:
    batch_size = config.parameters_static.batch_size
    num_classes = config.parameters.num_classes
    target_checkpoint_dir = config.paths.target_checkpoint_dir
    root_dir = Path(config.paths.current_root).parent
    model_name = config.parameters.model_arch

    target_model = load_highest_round_number_target_model(root_dir, target_checkpoint_dir, model_name, device)

    target_train_loader = DataLoader(target_train, batch_size=batch_size, shuffle=False, num_workers=2)

    target_test_loader = DataLoader(target_test, batch_size=batch_size, shuffle=False, num_workers=2)
    target_model.eval()

    # Extract features from the target model for members (train) and non-members (test)
    train_feats = extract_features_by_class(target_model, target_train_loader, label_indicator=1,
                                            num_classes=num_classes, device=device)
    test_feats = extract_features_by_class(target_model, target_test_loader, label_indicator=0, num_classes=num_classes,
                                           device=device)

    all_aucs = []
    all_f1s = []
    all_accs = []
    all_precisions = []
    all_recs = []
    all_fars = []

    per_class_metrics = defaultdict(dict)

    print(f"\n=== Evaluating Pooled Per-Class Attack Models ===")

    for cls in range(num_classes):
        if cls not in attack_models:
            print(f"⚠️ Skipping Class {cls} — no attack model trained.")
            continue

        train_cls_feats = train_feats.get(cls, [])
        test_cls_feats = test_feats.get(cls, [])

        if not train_cls_feats or not test_cls_feats:
            print(
                f"⚠️ Skipping Class {cls} — insufficient data: Train={len(train_cls_feats)}, Test={len(test_cls_feats)}")
            continue

        X_train, _ = zip(*train_cls_feats)
        X_test, _ = zip(*test_cls_feats)

        min_len = min(len(X_train), len(X_test))
        X_eval = np.vstack((X_train[:min_len], X_test[:min_len]))
        y_eval = np.array([1] * min_len + [0] * min_len)

        X_scaled = scalers[cls].transform(X_eval)
        y_pred = attack_models[cls].predict(X_scaled)
        y_scores = attack_models[cls].predict_proba(X_scaled)[:, 1]

        acc = accuracy_score(y_eval, y_pred)
        prec = precision_score(y_eval, y_pred, zero_division=0)
        rec = recall_score(y_eval, y_pred, zero_division=0)
        f1 = f1_score(y_eval, y_pred, zero_division=0)
        auc = roc_auc_score(y_eval, y_scores)

        tn, fp, fn, tp = confusion_matrix(y_eval, y_pred).ravel()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Store per-class metrics
        per_class_metrics[cls]['accuracy'] = acc
        per_class_metrics[cls]['precision'] = prec
        per_class_metrics[cls]['recall'] = rec
        per_class_metrics[cls]['f1_score'] = f1
        per_class_metrics[cls]['auc'] = auc
        per_class_metrics[cls]['far'] = far

        # Store for global averages
        all_accs.append(acc)
        all_precisions.append(prec)
        all_recs.append(rec)
        all_f1s.append(f1)
        all_aucs.append(auc)
        all_fars.append(far)

        print(f"Class {cls}: Acc={acc:.2f}, Prec={prec:.2f}, Rec={rec:.2f}, F1={f1:.2f}, AUC={auc:.2f}, FAR={far:.2f}")

    log_per_class_metrics(per_class_metrics)

    # === Per-Class Summary ===
    print(f"\n=== Per-Class Attack Metrics ===")
    for cls in sorted(per_class_metrics.keys()):
        metrics = per_class_metrics[cls]
        print(f"Class {cls}: Acc={metrics['accuracy']:.2f}, AUC={metrics['auc']:.2f}, F1={metrics['f1_score']:.2f}")

    # === Overall Summary ===
    overall_accuracy = np.mean(all_accs) if all_accs else 0
    overall_precision = np.mean(all_precisions) if all_precisions else 0
    overall_recall = np.mean(all_recs) if all_recs else 0
    overall_f1 = np.mean(all_f1s) if all_f1s else 0
    overall_auc = np.mean(all_aucs) if all_aucs else 0
    overall_far = np.mean(all_fars) if all_fars else 0

    overall_std_accuracy = np.std(all_accs) if all_accs else 0
    overall_std_precision = np.std(all_precisions) if all_precisions else 0
    overall_std_recall = np.std(all_recs) if all_recs else 0
    overall_std_f1 = np.std(all_f1s) if all_f1s else 0
    overall_std_auc = np.std(all_aucs) if all_aucs else 0
    overall_std_far = np.std(all_fars) if all_fars else 0

    print(f"\n=== Overall Attack Performance ===")
    print(f"Accuracy = {overall_accuracy:.2f} ± {overall_std_accuracy:.2f}")
    print(f"Precision = {overall_precision:.2f} ± {overall_std_precision:.2f}")
    print(f"Recall = {overall_recall:.2f} ± {overall_std_recall:.2f}")
    print(f"F1       = {overall_f1:.2f} ± {overall_std_f1:.2f}")
    print(f"AUC      = {overall_auc:.2f} ± {overall_std_auc:.2f}")
    print(f"FAR      = {overall_far:.2f} ± {overall_std_far:.2f}")

    log_overall_metrics_with_error_bars(
        accuracy=overall_accuracy, std_accuracy=overall_std_accuracy,
        precision=overall_precision, std_precision=overall_std_precision,
        recall=overall_recall, std_recall=overall_std_recall,
        f1=overall_f1, std_f1=overall_std_f1,
        auc=overall_auc, std_auc=overall_std_auc,
        far=overall_far, std_far=overall_std_far
    )


def load_highest_round_number_target_model(root_dir, target_checkpoint_dir, model_name, device) -> nn.Module:
    release_model(None, device.type)
    model = create_model(model_name)

    # Get all checkpoint paths
    checkpoint_paths = glob.glob(os.path.join(root_dir, target_checkpoint_dir, "global_model_round_*.pth"))

    if not checkpoint_paths:
        raise FileNotFoundError("No checkpoint files found.")

    # Extract round numbers and map to paths
    checkpoints_with_rounds = []
    for path in checkpoint_paths:
        match = re.search(r'global_model_round_(\d+)\.pth', os.path.basename(path))
        if match:
            round_number = int(match.group(1))
            checkpoints_with_rounds.append((round_number, path))

    if not checkpoints_with_rounds:
        raise ValueError("No valid checkpoint files with round numbers found.")

    # Get the path with the highest round number
    latest_round, latest_checkpoint_path = max(checkpoints_with_rounds, key=lambda x: x[0])

    model.load_state_dict(torch.load(latest_checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded target model from: {latest_checkpoint_path} (round {latest_round})")

    return model


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@hydra.main(version_base=None, config_path="./configs_mia", config_name="mia_run_base.yaml")
def main(config: MiaConfig):
    print(f"\n🚀 Running experiment with config: {config}\n")
    # Initialize wandb run
    wandb.init(project="mia-shadow-attack", config=OmegaConf.to_container(config, resolve=True), name="mia_run")

    seed_everything(config.parameters_static.seed)
    requested_device = config.parameters_static.device

    if "cuda" in requested_device and not torch.cuda.is_available():
        print("[FedCustom] ⚠️ CUDA requested but not available. Falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    shadow_train, target_train, target_test, shadow_test = load_data(config=config)

    shadow_train_subsets, shadow_test_subsets = sample_shadow_datasets_with_overlap(
        shadow_train_dataset=shadow_train,
        shadow_test_dataset=shadow_test,
        config=config)

    per_shadow_per_class_data = train_all_shadow_models_and_collect_features(
        shadow_train_subsets=shadow_train_subsets,
        shadow_test_subsets=shadow_test_subsets,
        config=config, device=device)

    attack_models, scalers = train_per_class_attack_models(per_shadow_per_class_data=per_shadow_per_class_data,
                                                           config=config)

    evaluate_attack_models(attack_models=attack_models, scalers=scalers, target_train=target_train,
                           target_test=target_test, config=config, device=device)

    # Finish the wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
