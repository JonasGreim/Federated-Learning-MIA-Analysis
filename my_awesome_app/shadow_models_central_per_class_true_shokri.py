import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor
import numpy as np
import os
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils import resample
from my_awesome_app.models.mia_paper_target_shadow_model import SimpleCNN
from my_awesome_app.task import get_transforms_custom, seed_everything, seed_worker, release_model
from collections import defaultdict

# === Configuration ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
BATCH_SIZE = 32
SHADOW_EPOCHS = 100
LEARNING_RATE = 0.05
NUM_CLASSES = 10
NUM_SHADOW_MODELS = 4
USE_EXTRA_ATTACK_FEATURES = False
MODEL_ARCH = SimpleCNN
TARGET_CHECKPOINT_DIR = "../model_checkpoints_target"
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
SPLIT_DIR = os.path.join(ROOT_DIR, "splits")

seed_everything(SEED)
g = torch.Generator().manual_seed(SEED)


# === Utility Functions ===
def get_transforms():
    return Compose([
        ToTensor(),
        # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def load_data():
    train_dataset = CIFAR10(root=DATA_DIR, train=True, download=True, transform=get_transforms_custom())
    test_dataset = CIFAR10(root=DATA_DIR, train=False, download=True, transform=get_transforms_custom())

    D1 = np.load(os.path.join(SPLIT_DIR, "D1_indices.npy")).tolist()
    D2 = np.load(os.path.join(SPLIT_DIR, "D2_indices.npy")).tolist()
    D3 = np.load(os.path.join(SPLIT_DIR, "D3_indices.npy")).tolist()
    D4 = np.load(os.path.join(SPLIT_DIR, "D4_indices.npy")).tolist()

    target_train = Subset(train_dataset, D1)
    target_test = Subset(train_dataset, D2)
    shadow_train = Subset(train_dataset, D3)
    shadow_test = Subset(train_dataset, D4)

    return shadow_train, target_train, target_test, shadow_test


def train_model(model, dataloader, epochs):
    model = model.to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
    return model


def stratified_shadow_splits(shadow_dataset, num_shadow_models, seed):
    # Extract targets from the original dataset
    targets = np.array([shadow_dataset.dataset.targets[i] for i in shadow_dataset.indices])

    # Group indices by class
    class_to_indices = defaultdict(list)
    for idx, label in zip(shadow_dataset.indices, targets):
        class_to_indices[label].append(idx)

    # Shuffle each class
    rng = np.random.RandomState(seed)
    for cls in class_to_indices:
        rng.shuffle(class_to_indices[cls])

    # Split per class into equal parts
    per_class_splits = {cls: np.array_split(class_to_indices[cls], num_shadow_models) for cls in class_to_indices}

    # Build shadow subsets
    shadow_splits = []
    for i in range(num_shadow_models):
        indices = []
        for cls in per_class_splits:
            indices.extend(per_class_splits[cls][i])
        rng.shuffle(indices)  # Shuffle to mix classes
        shadow_splits.append(Subset(shadow_dataset.dataset, indices))

    return shadow_splits


def load_shadow_data_with_test_splits(num_shadow_models, seed, g, shadow_train_dataset, shadow_test_dataset):
    # Split shadow_train_dataset into disjoint subsets
    shadow_train_subsets = stratified_shadow_splits(shadow_train, num_shadow_models, seed)
    shadow_test_subsets = stratified_shadow_splits(shadow_test, num_shadow_models, seed)

    return shadow_train_subsets, shadow_test_subsets


def train_all_shadow_models_and_collect_features(shadow_train_subsets, shadow_test_subsets, num_classes, model_arch,
                                                 shadow_epochs, batch_size, device, use_extra_features, g):
    per_shadow_per_class_data = []

    for i, (train_subset, test_subset) in enumerate(zip(shadow_train_subsets, shadow_test_subsets)):
        print(f"🔄 Training Shadow Model {i + 1}/{len(shadow_train_subsets)}")
        model = model_arch()

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker,
                                  generator=g)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker,
                                 generator=g)

        model = train_model(model, train_loader, shadow_epochs)

        member_features = extract_features_by_class(model, train_loader, label_indicator=1,
                                                    use_extra=use_extra_features)
        nonmember_features = extract_features_by_class(model, test_loader, label_indicator=0,
                                                       use_extra=use_extra_features)

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


def train_per_shadow_per_class_attack_models(per_shadow_per_class_data, num_classes, seed):
    attack_models = []
    scalers = []

    for shadow_idx, class_data in enumerate(per_shadow_per_class_data):
        shadow_attack_models = {}
        shadow_scalers = {}

        for cls in range(num_classes):
            member = class_data[cls]['member']
            nonmember = class_data[cls]['nonmember']

            if not member or not nonmember:
                continue

            X_member, y_member = zip(*member)
            X_nonmember, y_nonmember = zip(*nonmember)

            # Oversampling = ensure balanced attack model datasets
            max_len = max(len(X_member), len(X_nonmember))

            X_member, y_member = resample(X_member, y_member, n_samples=max_len, random_state=seed, replace=True)
            X_nonmember, y_nonmember = resample(X_nonmember, y_nonmember, n_samples=max_len, random_state=seed,
                                                replace=True)

            X = np.vstack((X_member, X_nonmember))
            y = np.array(y_member + y_nonmember)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=seed)
            clf.fit(X_scaled, y)

            shadow_attack_models[cls] = clf
            shadow_scalers[cls] = scaler

        attack_models.append(shadow_attack_models)
        scalers.append(shadow_scalers)

    return attack_models, scalers


def evaluate_all_attack_models(attack_models, scalers, target_model, target_train_loader, target_test_loader,
                               num_classes, use_extra_features):
    target_model.eval()

    train_feats = extract_features_by_class(target_model, target_train_loader, label_indicator=1,
                                            use_extra=use_extra_features)
    test_feats = extract_features_by_class(target_model, target_test_loader, label_indicator=0,
                                           use_extra=use_extra_features)

    all_aucs = []
    all_f1s = []

    for shadow_idx, (attack_model_dict, scaler_dict) in enumerate(zip(attack_models, scalers)):
        print(f"\n=== Evaluating Attack Models from Shadow {shadow_idx + 1} ===")

        for cls in range(num_classes):
            if cls not in attack_model_dict:
                continue

            train_cls_feats = train_feats.get(cls, [])
            test_cls_feats = test_feats.get(cls, [])

            if not train_cls_feats or not test_cls_feats:
                continue

            X_train, _ = zip(*train_cls_feats)
            X_test, _ = zip(*test_cls_feats)

            min_len = min(len(X_train), len(X_test))
            X_eval = np.vstack((X_train[:min_len], X_test[:min_len]))
            y_eval = np.array([1] * min_len + [0] * min_len)

            X_scaled = scaler_dict[cls].transform(X_eval)
            y_pred = attack_model_dict[cls].predict(X_scaled)
            y_scores = attack_model_dict[cls].predict_proba(X_scaled)[:, 1]

            tn, fp, fn, tp = confusion_matrix(y_eval, y_pred).ravel()
            far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            all_aucs.append(roc_auc_score(y_eval, y_scores))
            all_f1s.append(f1_score(y_eval, y_pred))

            print(
                f"Class {cls}: Acc={accuracy_score(y_eval, y_pred):.2f}, Prec={precision_score(y_eval, y_pred):.2f}, Rec={recall_score(y_eval, y_pred):.2f}, F1={f1_score(y_eval, y_pred):.2f}, AUC={roc_auc_score(y_eval, y_scores):.2f}, FAR={far:.2f}")
    print(
        f"\n=== Overall Attack Performance: Avg AUC = {np.mean(all_aucs):.2f}, Avg F1 = {np.mean(all_f1s):.2f}")


def extract_features_by_class(model, dataloader, label_indicator, use_extra=False):
    model.eval()
    class_features = defaultdict(list)
    eps = 1e-10

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)

            # Compute predicted classes (not part original shokri code -> improvement of other papers)
            # pred_classes = outputs.argmax(dim=1)
            # Build mask for correct predictions
            # mask = pred_classes == labels

            if use_extra:
                entropy = (-probs * (probs + eps).log()).sum(dim=1, keepdim=True)
                top2 = probs.topk(2, dim=1).values
                margin = (top2[:, 0] - top2[:, 1]).unsqueeze(1)
                probs = torch.cat([probs, entropy, margin], dim=1)

            probs_np = probs.cpu().numpy()
            labels_np = labels.cpu().numpy()

            for i in range(labels_np.shape[0]):
                class_idx = labels_np[i]
                class_features[class_idx].append((probs_np[i], label_indicator))

    return class_features


def load_target_model():
    release_model(None, DEVICE.type)
    model = MODEL_ARCH()
    ckpt_path = max(glob.glob(f"{TARGET_CHECKPOINT_DIR}/global_model_round_*"), key=os.path.getctime)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    print(f"✅ Loaded target model: {ckpt_path}")
    return model


# === Main ===
if __name__ == "__main__":
    shadow_train, target_train, target_test, shadow_test = load_data()

    shadow_train_subsets, shadow_test_subsets = load_shadow_data_with_test_splits(NUM_SHADOW_MODELS, SEED, g,
                                                                                  shadow_train, shadow_test)

    per_shadow_per_class_data = train_all_shadow_models_and_collect_features(
        shadow_train_subsets, shadow_test_subsets, NUM_CLASSES, MODEL_ARCH, SHADOW_EPOCHS, BATCH_SIZE, DEVICE,
        USE_EXTRA_ATTACK_FEATURES, g
    )

    attack_models, scalers = train_per_shadow_per_class_attack_models(per_shadow_per_class_data, NUM_CLASSES, SEED)

    target_model = load_target_model()
    target_train_loader = DataLoader(target_train, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker,
                                     generator=g)
    target_test_loader = DataLoader(target_test, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker,
                                    generator=g)

    evaluate_all_attack_models(attack_models, scalers, target_model, target_train_loader, target_test_loader,
                               NUM_CLASSES, USE_EXTRA_ATTACK_FEATURES)
