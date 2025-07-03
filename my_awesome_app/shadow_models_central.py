from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from my_awesome_app.models.mia_paper_target_shadow_model import SimpleCNN
from my_awesome_app.models.simple_model import NetSimple
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import glob
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.metrics import confusion_matrix
import random
from sklearn.utils import resample
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset
from my_awesome_app.task import get_transforms_custom, seed_everything, seed_worker

# === Config ===
USE_EXTRA_ATTACK_FEATURES = True  # use extra features for attack model (entropy, margin)
SHADOW_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.05  # learning rate for shadow models
TARGET_CHECKPOINT_DIR = "../model_checkpoints_target"
MODEL_Arch = SimpleCNN
NUM_SHADOW_MODELS = 5  # number of shadow models to train
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
SPLIT_DIR = os.path.join(ROOT_DIR, "splits")
DEVICE_STR = "cuda"
SEED = 42

# Generate a list of shadow model architectures
SHADOW_MODEL_ARCHS = [MODEL_Arch] * NUM_SHADOW_MODELS  # You could use here different architectures for each shadow model

# Seed everything for reproducibility
seed_everything(SEED)
g = torch.Generator().manual_seed(SEED)

# Set device
DEVICE = torch.device(DEVICE_STR if torch.cuda.is_available() or "cpu" in DEVICE_STR else "cpu")
print(f"Using device: {DEVICE} {'✅ GPU available' if DEVICE.type == 'cuda' else '⚠️ CPU only'}")


# === Utility Functions ===

def get_transforms():
    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def apply_transforms(batch):
        batch["img"] = [transform(img) for img in batch["img"]]
        return batch

    return apply_transforms


def load_and_prepare_data():
    # Load full CIFAR-10 train and test sets
    train_dataset = CIFAR10(root=DATA_DIR, train=True, download=True, transform=get_transforms_custom())
    test_dataset = CIFAR10(root=DATA_DIR, train=False, download=True, transform=get_transforms_custom())

    # Load predefined split indices
    D1 = np.load(os.path.join(SPLIT_DIR, "D1_indices.npy")).tolist()
    D2 = np.load(os.path.join(SPLIT_DIR, "D2_indices.npy")).tolist()
    D3 = np.load(os.path.join(SPLIT_DIR, "D3_indices.npy")).tolist()
    D4 = np.load(os.path.join(SPLIT_DIR, "D4_indices.npy")).tolist()

    # Create Subsets
    target_train_dataset = Subset(train_dataset, D1)
    target_test_dataset = Subset(train_dataset, D2)

    shadow_train_dataset = Subset(train_dataset, D3)
    shadow_test_dataset = Subset(train_dataset, D4)

    # split shadow_train_dataset across multiple shadow models
    shadow_split_size = len(shadow_train_dataset) // NUM_SHADOW_MODELS
    g = torch.Generator().manual_seed(SEED)
    shadow_subsets = random_split(
        shadow_train_dataset,
        [shadow_split_size] * (NUM_SHADOW_MODELS - 1) + [
            len(shadow_train_dataset) - shadow_split_size * (NUM_SHADOW_MODELS - 1)
        ],
        generator=g
    )

    return shadow_subsets, target_train_dataset, target_test_dataset, shadow_test_dataset


def train_model(model, dataloader, epochs):
    model = model.to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for _ in range(epochs):
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

    return model


def extract_attack_features(model, dataloader, label, use_extra_features):
    model.eval()
    features, labels = [], []
    eps = 1e-10

    with torch.no_grad():
        for batch in dataloader:
            inputs, _ = batch
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)  # you could use the logits -> not realistic for black box
            probs = F.softmax(outputs, dim=1)  # use softmax to get probabilities

            if use_extra_features:
                entropy = (-probs * (probs + eps).log()).sum(dim=1, keepdim=True)
                top2 = probs.topk(2, dim=1).values
                margin = (top2[:, 0] - top2[:, 1]).unsqueeze(1)
                probs = torch.cat([probs, entropy, margin], dim=1)

            features.extend(probs.cpu().numpy())
            labels.extend([label] * probs.size(0))

    return features, labels


def evaluate_shadow_model(shadow_model, dataloader):
    shadow_model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = shadow_model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / len(dataloader)
    return avg_loss, correct / total


def train_shadow_models(shadow_subsets, test_dataset):
    all_member_feats, all_member_labels = [], []
    all_nonmember_feats, all_nonmember_labels = [], []

    for i, (subset, arch) in enumerate(zip(shadow_subsets, SHADOW_MODEL_ARCHS)):
        print(f"Training Shadow Model {i + 1}/{NUM_SHADOW_MODELS} with {arch.__name__}")
        model = arch()
        loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=seed_worker, generator=g)
        model = train_model(model, loader, SHADOW_EPOCHS)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, worker_init_fn=seed_worker, generator=g)

        # === Add logging here ===
        train_loss, train_acc = evaluate_shadow_model(model, loader)
        test_loss, test_acc = evaluate_shadow_model(model, test_loader)
        print(
            f"✅ Shadow Model {i + 1} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2%}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2%}")

        member_feats, member_labels = extract_attack_features(model, loader, 1, USE_EXTRA_ATTACK_FEATURES)
        nonmember_feats, nonmember_labels = extract_attack_features(model,
                                                                    DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                                                               worker_init_fn=seed_worker, generator=g),
                                                                    0,
                                                                    USE_EXTRA_ATTACK_FEATURES)

        all_member_feats.extend(member_feats)
        all_member_labels.extend(member_labels)
        all_nonmember_feats.extend(nonmember_feats)
        all_nonmember_labels.extend(nonmember_labels)

    return all_member_feats, all_member_labels, all_nonmember_feats, all_nonmember_labels


def train_attack_model(member_feats, member_labels, nonmember_feats, nonmember_labels):
    print("✅ Member count (train attack model):", len(member_feats))
    print("✅ Non-member count (train attack model):", len(nonmember_feats))

    # === Balance the classes
    min_len = min(len(member_feats), len(nonmember_feats))

    member_feats, member_labels = resample(
        member_feats, member_labels, n_samples=min_len, random_state=42, replace=False
    )
    nonmember_feats, nonmember_labels = resample(
        nonmember_feats, nonmember_labels, n_samples=min_len, random_state=42, replace=False
    )
    print("✅ Member/Non-member count after balancing:", min_len)

    X = np.vstack((member_feats, nonmember_feats))
    y = np.array(member_labels + nonmember_labels)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler, y


def load_latest_target_model():
    model = MODEL_Arch()
    model_checkpoint_path = max(glob.glob(f"{TARGET_CHECKPOINT_DIR}/global_model_round_*"), key=os.path.getctime)
    model.load_state_dict(torch.load(model_checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("Loaded target model from:", model_checkpoint_path)
    return model


def compute_far(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn)


def evaluate_attack_model(attack_model, scaler, target_model, target_train_loader, test_loader):
    member_feats, _ = extract_attack_features(target_model, target_train_loader, 1, USE_EXTRA_ATTACK_FEATURES)
    nonmember_feats, _ = extract_attack_features(target_model, test_loader, 0, USE_EXTRA_ATTACK_FEATURES)

    # balance the number of member and non-member samples for evaluation
    min_len = min(len(member_feats), len(nonmember_feats))
    random.seed(42)
    random.shuffle(member_feats)
    random.shuffle(nonmember_feats)
    member_feats = member_feats[:min_len]
    nonmember_feats = nonmember_feats[:min_len]

    X = np.vstack((member_feats, nonmember_feats))
    y_true = np.array([1] * len(member_feats) + [0] * len(nonmember_feats))
    X_scaled = scaler.transform(X)

    # Predict
    y_scores = attack_model.predict_proba(X_scaled)[:, 1]  # Probabilities
    y_pred = attack_model.predict(X_scaled)  # Hard predictions
    far = compute_far(y_true, y_pred)

    print("\n=== Attack Model Evaluation ===")
    print(
        f"Accuracy : {accuracy_score(y_true, y_pred):.2f}")  # (TP+TN)/(TP+TN+FP+FN) how many were correctly classified out of all samples
    print(
        f"Precision: {precision_score(y_true, y_pred):.2f}")  # TP / (TP + FP) predicted as positive, how many were actually positive?
    print(
        f"Recall   : {recall_score(y_true, y_pred):.2f}")  # TP / (TP + FN) how many members were correctly identified out of all members
    print(f"F1 Score : {f1_score(y_true, y_pred):.2f}")
    print(f"AUC      : {roc_auc_score(y_true, y_scores):.2f}")  # TPR (Recall) vs FPR (0.5 = random guessing)
    print(
        f"False Alarm Rate (FAR): {far:.2f}")  # FP / (FP/TN) rate proportion of non-member samples that are incorrectly classified as members (<10% & high recall -> good)
    print("Class distribution in test attack data:", Counter(y_true))


# === Main Script ===
if __name__ == "__main__":
    shadow_subsets, target_train_dataset, target_test_dataset, shadow_test_dataset = load_and_prepare_data()

    member_feats, member_labels, nonmember_feats, nonmember_labels = train_shadow_models(shadow_subsets,
                                                                                         shadow_test_dataset)
    attack_model, scaler, y_attack = train_attack_model(member_feats, member_labels, nonmember_feats, nonmember_labels)

    target_model = load_latest_target_model()
    target_train_loader = DataLoader(target_train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                     worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(target_test_dataset, batch_size=BATCH_SIZE, worker_init_fn=seed_worker, generator=g)

    evaluate_attack_model(attack_model, scaler, target_model, target_train_loader, test_loader)
