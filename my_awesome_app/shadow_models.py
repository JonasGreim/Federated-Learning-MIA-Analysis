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

# settings
USE_EXTRA_ATTACK_FEATURES = True  # use extra features for attack model (entropy, margin)
NUM_SHADOW_MODELS = 5
SHADOW_EPOCHS = 1
BATCH_SIZE = 32
CHECKPOINT_DIR = "../model_checkpoints"

# === Load CIFAR-10 dataset ===
dataset = load_dataset('cifar10', batch_size=32)


# === Transform function ===
def get_transforms():
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    return apply_transforms


dataset = dataset.with_transform(get_transforms())

# === Dataset Splits ===
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# split train dataset into shadow and target datasets (50:50)
train_split_size = len(train_dataset) // 2
shadow_train_dataset, target_train_dataset = random_split(train_dataset, [train_split_size, train_split_size])

# split shadow dataset into NUM_SHADOW_MODELS parts
shadow_split_size = len(shadow_train_dataset) // NUM_SHADOW_MODELS

# Split shadow training data into multiple parts
shadow_subsets = torch.utils.data.random_split(
    shadow_train_dataset,
    [shadow_split_size] * (NUM_SHADOW_MODELS - 1) + [len(shadow_train_dataset) - shadow_split_size * (NUM_SHADOW_MODELS - 1)]
)

# load data into dataloader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
target_train_loader = DataLoader(target_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# === Train Shadow Model ===
def train_shadow_model(shadow_model, dataloader, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shadow_model = shadow_model.to(device)
    optimizer = torch.optim.Adam(shadow_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    shadow_model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            inputs, labels = batch["img"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = shadow_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return shadow_model


# === Extract Features for Attack Model ===
# Collect Confidence Scores (for attack model)
# We'll use softmax outputs as input features and whether the sample was in the training set as the label.
# label=1 for members, label=0 for non-members
def get_attack_features(target_model, dataloader, label, use_extra_features=False):
    target_model.eval()
    device = next(target_model.parameters()).device
    features = []
    labels = []
    eps = 1e-10  # for safe log

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["img"].to(device)
            outputs = target_model(inputs)
            probs = F.softmax(outputs, dim=1)  # shape: [batch_size, 10]

            batch_features = probs

            if use_extra_features:
                # Entropy
                entropy = (-probs * (probs + eps).log()).sum(dim=1).unsqueeze(1)
                # Margin (top-1 - top-2)
                top2 = probs.topk(2, dim=1)[0]
                margin = (top2[:, 0] - top2[:, 1]).unsqueeze(1)
                # Concatenate extra features
                batch_features = torch.cat([probs, entropy, margin], dim=1)

            features.extend(batch_features.cpu().numpy())
            labels.extend([label] * batch_features.size(0))

    return features, labels


shadow_models = []
all_shadow_member_features = []
all_shadow_member_labels = []
all_shadow_nonmember_features = []
all_shadow_nonmember_labels = []

# === Shadow Training Function ===
# Train each shadow model and collect features
for i, subset in enumerate(shadow_subsets):
    print(f"\n--- Training Shadow Model {i + 1}/{NUM_SHADOW_MODELS} ---")
    model = NetSimple()
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
    model = train_shadow_model(model, loader, epochs=SHADOW_EPOCHS)

    shadow_models.append(model)

    # Member features (from model's own training data)
    member_features, member_labels = get_attack_features(model, loader, label=1, use_extra_features=USE_EXTRA_ATTACK_FEATURES)
    # Non-member features (from shared test data)
    nonmember_features, nonmember_labels = get_attack_features(model, test_loader, label=0, use_extra_features=USE_EXTRA_ATTACK_FEATURES)

    # Balance and collect
    min_len = min(len(member_features), len(nonmember_features))
    all_shadow_member_features.extend(member_features[:min_len])
    all_shadow_member_labels.extend(member_labels[:min_len])
    all_shadow_nonmember_features.extend(nonmember_features[:min_len])
    all_shadow_nonmember_labels.extend(nonmember_labels[:min_len])

# Combine all shadow model features
X_attack = np.vstack((all_shadow_member_features, all_shadow_nonmember_features))
y_attack = np.array(all_shadow_member_labels + all_shadow_nonmember_labels)

# Normalize and train attack model
scaler = StandardScaler()
X_attack_scaled = scaler.fit_transform(X_attack)
attack_model = GradientBoostingClassifier().fit(X_attack_scaled, y_attack)

# Load target model
list_of_files = [fname for fname in glob.glob(f"{CHECKPOINT_DIR}/global_model_round_*")]
latest_round_file = max(list_of_files, key=os.path.getctime)

target_model = NetSimple()
target_model.load_state_dict(torch.load(f=latest_round_file, map_location='cpu'))
print("Loading pre-trained model from: ", latest_round_file)
target_model.eval()

# Run the Attack on Target Mode
# Measured how well it could infer membership

target_member_features, _ = get_attack_features(target_model, target_train_loader, 1, use_extra_features=USE_EXTRA_ATTACK_FEATURES)
target_nonmember_features, _ = get_attack_features(target_model, test_loader, 0, use_extra_features=USE_EXTRA_ATTACK_FEATURES)

X_target_attack = np.vstack((target_member_features, target_nonmember_features))  # multi arrays stacked to (1,N)
X_target_attack_scaled = scaler.transform(X_target_attack)
# ground truth
y_target_true = np.array([1] * len(target_member_features) + [0] * len(target_nonmember_features))

y_pred = attack_model.predict(X_target_attack_scaled)

# evaluate attack: compare to ground truth
attack_accuracy = accuracy_score(y_target_true, y_pred)
precision = precision_score(y_target_true, y_pred)
recall = recall_score(y_target_true, y_pred)
f1 = f1_score(y_target_true, y_pred)
auc = roc_auc_score(y_target_true, y_pred)

# Attack Accuracy: 0.71 (10 or 20 rounds) -> baseline random guessing -> 50% -> 50% non-members und 50% members
print("=== Attack Model Evaluation ===")
print(f"Attack Accuracy: {attack_accuracy:.2f}")
print(f"Precision: {precision:.2f}")  # How many of your positive predictions were correct
print(f"Recall   : {recall:.2f}")  # How many actual members did you find?
print(f"F1 Score : {f1:.2f}")  # Balance between precision and recall
print(f"AUC      : {auc:.2f}")  # 0.5 -> random guessing, 1 perfect, Measures the model’s ability to separate the classes across all possible thresholds
print("Class distribution in attack data:", Counter(y_attack))
