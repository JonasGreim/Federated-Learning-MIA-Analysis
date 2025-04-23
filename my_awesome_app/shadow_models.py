from datasets import load_dataset
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from my_awesome_app.models.simple_model import NetSimple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
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
from multiprocessing import Pool, cpu_count, set_start_method
import warnings

# === Config ===
USE_EXTRA_ATTACK_FEATURES = True  # use extra features for attack model (entropy, margin)
NUM_SHADOW_MODELS = 5
SHADOW_EPOCHS = 1
BATCH_SIZE = 32
CHECKPOINT_DIR = "../model_checkpoints"


# === Utility Functions ===

def get_transforms():
    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def apply_transforms(batch):
        batch["img"] = [transform(img) for img in batch["img"]]
        return batch

    return apply_transforms


def load_and_prepare_data():
    dataset = load_dataset('cifar10', batch_size=BATCH_SIZE).with_transform(get_transforms())
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    half_size = len(train_dataset) // 2
    shadow_train_dataset, target_train_dataset = random_split(train_dataset, [half_size, len(train_dataset) - half_size])

    shadow_split_size = len(shadow_train_dataset) // NUM_SHADOW_MODELS
    shadow_subsets = random_split(
        shadow_train_dataset,
        [shadow_split_size] * (NUM_SHADOW_MODELS - 1) + [len(shadow_train_dataset) - shadow_split_size * (NUM_SHADOW_MODELS - 1)]
    )

    return shadow_subsets, target_train_dataset, test_dataset


def train_model(model, dataloader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for _ in range(epochs):
        for batch in dataloader:
            inputs, labels = batch["img"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

    return model


def extract_attack_features(model, dataloader, label, use_extra_features):
    model.eval()
    device = next(model.parameters()).device
    features, labels = [], []
    eps = 1e-10

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["img"].to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)

            if use_extra_features:
                entropy = (-probs * (probs + eps).log()).sum(dim=1, keepdim=True)
                top2 = probs.topk(2, dim=1).values
                margin = (top2[:, 0] - top2[:, 1]).unsqueeze(1)
                probs = torch.cat([probs, entropy, margin], dim=1)

            features.extend(probs.cpu().numpy())
            labels.extend([label] * probs.size(0))

    return features, labels


def train_shadow_models(shadow_subsets, test_dataset):
    all_member_feats, all_member_labels = [], []
    all_nonmember_feats, all_nonmember_labels = [], []

    for i, subset in enumerate(shadow_subsets):
        print(f"Training Shadow Model {i + 1}/{NUM_SHADOW_MODELS}")
        model = NetSimple()
        loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
        model = train_model(model, loader, SHADOW_EPOCHS)

        member_feats, member_labels = extract_attack_features(model, loader, 1, USE_EXTRA_ATTACK_FEATURES)
        nonmember_feats, nonmember_labels = extract_attack_features(model, DataLoader(test_dataset, batch_size=BATCH_SIZE), 0, USE_EXTRA_ATTACK_FEATURES)

        min_len = min(len(member_feats), len(nonmember_feats))
        all_member_feats.extend(member_feats[:min_len])
        all_member_labels.extend(member_labels[:min_len])
        all_nonmember_feats.extend(nonmember_feats[:min_len])
        all_nonmember_labels.extend(nonmember_labels[:min_len])

    return all_member_feats, all_member_labels, all_nonmember_feats, all_nonmember_labels


def train_attack_model(member_feats, member_labels, nonmember_feats, nonmember_labels):
    X = np.vstack((member_feats, nonmember_feats))
    y = np.array(member_labels + nonmember_labels)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = GradientBoostingClassifier().fit(X_scaled, y)
    return model, scaler, y


def load_latest_target_model():
    model = NetSimple()
    model_path = max(glob.glob(f"{CHECKPOINT_DIR}/global_model_round_*"), key=os.path.getctime)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print("Loaded target model from:", model_path)
    return model


def evaluate_attack_model(attack_model, scaler, target_model, target_train_loader, test_loader):
    member_feats, _ = extract_attack_features(target_model, target_train_loader, 1, USE_EXTRA_ATTACK_FEATURES)
    nonmember_feats, _ = extract_attack_features(target_model, test_loader, 0, USE_EXTRA_ATTACK_FEATURES)

    X = np.vstack((member_feats, nonmember_feats))
    y_true = np.array([1] * len(member_feats) + [0] * len(nonmember_feats))
    y_pred = attack_model.predict(scaler.transform(X))

    print("\n=== Attack Model Evaluation ===")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.2f}")
    print(f"Precision: {precision_score(y_true, y_pred):.2f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.2f}")
    print(f"F1 Score : {f1_score(y_true, y_pred):.2f}")
    print(f"AUC      : {roc_auc_score(y_true, y_pred):.2f}")
    print("Class distribution in attack data:", Counter(y_true))


# === Main Script ===
if __name__ == "__main__":
    shadow_subsets, target_train_dataset, test_dataset = load_and_prepare_data()

    member_feats, member_labels, nonmember_feats, nonmember_labels = train_shadow_models(shadow_subsets, test_dataset)
    attack_model, scaler, y_attack = train_attack_model(member_feats, member_labels, nonmember_feats, nonmember_labels)

    target_model = load_latest_target_model()
    target_train_loader = DataLoader(target_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    evaluate_attack_model(attack_model, scaler, target_model, target_train_loader, test_loader)
