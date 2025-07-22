# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Subset, random_split
# from torchvision.datasets import CIFAR10
# from torchvision.transforms import Compose, ToTensor
# import numpy as np
# import os
# import glob
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
# from sklearn.utils import resample
# from flower.models.mia_paper_target_shadow_model import SimpleCNN
# from flower.task import get_transforms_custom, seed_everything, seed_worker, release_model
# import re
# from collections import defaultdict
#
#
# # === Configuration ===
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# SEED = 42
# BATCH_SIZE = 32
# SHADOW_EPOCHS = 100
# LEARNING_RATE = 0.05
# NUM_CLASSES = 10
# NUM_SHADOW_MODELS = 1
# USE_EXTRA_ATTACK_FEATURES = False
# MODEL_ARCH = SimpleCNN
# TARGET_CHECKPOINT_DIR = "../model_checkpoints_target"
# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# DATA_DIR = os.path.join(ROOT_DIR, "data")
# SPLIT_DIR = os.path.join(ROOT_DIR, "splits")
#
# seed_everything(SEED)
# g = torch.Generator().manual_seed(SEED)
#
#
# # === Utility Functions ===
# def get_transforms():
#     return Compose([
#         ToTensor(),
#         # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#
#
# def load_data():
#     train_dataset = CIFAR10(root=DATA_DIR, train=True, download=True, transform=get_transforms_custom())
#     test_dataset = CIFAR10(root=DATA_DIR, train=False, download=True, transform=get_transforms_custom())
#
#     D1 = np.load(os.path.join(SPLIT_DIR, "D1_indices.npy")).tolist()
#     D2 = np.load(os.path.join(SPLIT_DIR, "D2_indices.npy")).tolist()
#     D3 = np.load(os.path.join(SPLIT_DIR, "D3_indices.npy")).tolist()
#     D4 = np.load(os.path.join(SPLIT_DIR, "D4_indices.npy")).tolist()
#
#     target_train = Subset(train_dataset, D1)
#     target_test = Subset(train_dataset, D2)
#     shadow_train = Subset(train_dataset, D3)
#     shadow_test = Subset(train_dataset, D4)
#
#     shadow_split_size = len(shadow_train) // NUM_SHADOW_MODELS
#     shadow_subsets = random_split(shadow_train, [shadow_split_size] * (NUM_SHADOW_MODELS - 1) + [len(shadow_train) - shadow_split_size * (NUM_SHADOW_MODELS - 1)], generator=g)
#
#     return shadow_subsets, target_train, target_test, shadow_test
#
#
# def train_model(model, dataloader, epochs):
#     model = model.to(DEVICE)
#     optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
#     criterion = nn.CrossEntropyLoss()
#     model.train()
#     for _ in range(epochs):
#         for inputs, labels in dataloader:
#             inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
#             optimizer.zero_grad()
#             loss = criterion(model(inputs), labels)
#             loss.backward()
#             optimizer.step()
#     return model
#
#
# def extract_features_by_class(model, dataloader, label_indicator, use_extra=False):
#     model.eval()
#     class_features = defaultdict(list)
#     eps = 1e-10
#
#     with torch.no_grad():
#         for inputs, labels in dataloader:
#             inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
#             outputs = model(inputs)
#             probs = F.softmax(outputs, dim=1)
#
#             if use_extra:
#                 entropy = (-probs * (probs + eps).log()).sum(dim=1, keepdim=True)
#                 top2 = probs.topk(2, dim=1).values
#                 margin = (top2[:, 0] - top2[:, 1]).unsqueeze(1)
#                 probs = torch.cat([probs, entropy, margin], dim=1)
#
#             probs_np = probs.cpu().numpy()
#             labels_np = labels.cpu().numpy()
#
#             for i in range(labels_np.shape[0]):
#                 class_idx = labels_np[i]
#                 class_features[class_idx].append((probs_np[i], label_indicator))
#
#     return class_features
#
#
# def train_shadow_models(shadow_subsets, shadow_test_dataset):
#     per_class_data = defaultdict(lambda: {'member': [], 'nonmember': []})
#
#     for i, subset in enumerate(shadow_subsets):
#         model = MODEL_ARCH()
#         loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=seed_worker, generator=g)
#         model = train_model(model, loader, SHADOW_EPOCHS)
#
#         test_loader = DataLoader(shadow_test_dataset, batch_size=BATCH_SIZE, worker_init_fn=seed_worker, generator=g)
#
#         member_data = extract_features_by_class(model, loader, label_indicator=1, use_extra=USE_EXTRA_ATTACK_FEATURES)
#         nonmember_data = extract_features_by_class(model, test_loader, label_indicator=0, use_extra=USE_EXTRA_ATTACK_FEATURES)
#
#         for cls in range(NUM_CLASSES):
#             per_class_data[cls]['member'].extend([x for x in member_data.get(cls, [])])
#             per_class_data[cls]['nonmember'].extend([x for x in nonmember_data.get(cls, [])])
#
#         release_model(model, DEVICE.type)
#     return per_class_data
#
#
# def train_attack_models(per_class_data):
#     attack_models = {}
#     scalers = {}
#
#     for cls in range(NUM_CLASSES):
#         member = per_class_data[cls]['member']
#         nonmember = per_class_data[cls]['nonmember']
#
#         print(f"Class {cls}: Members={len(member)}, Non-Members={len(nonmember)}")
#
#         if not member or not nonmember:
#             print(f"⚠️ Skipping class {cls}: insufficient data (member: {len(member)}, non-member: {len(nonmember)})")
#             continue
#
#         X_member, y_member = zip(*member)
#         X_nonmember, y_nonmember = zip(*nonmember)
#
#         min_len = min(len(X_member), len(X_nonmember))
#         X_member, y_member = resample(X_member, y_member, n_samples=min_len, random_state=SEED, replace=False)
#         X_nonmember, y_nonmember = resample(X_nonmember, y_nonmember, n_samples=min_len, random_state=SEED, replace=False)
#
#         X = np.vstack((X_member, X_nonmember))
#         y = np.array(y_member + y_nonmember)
#
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)
#
#         clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=SEED)
#         clf.fit(X_scaled, y)
#
#         attack_models[cls] = clf
#         scalers[cls] = scaler
#
#     return attack_models, scalers
#
#
# def evaluate_per_class_attack_models(attack_models, scalers, target_model, target_train_loader, target_test_loader):
#     target_model.eval()
#     print("\n=== Per-Class Attack Model Evaluation ===")
#
#     train_features = extract_features_by_class(target_model, target_train_loader, label_indicator=1, use_extra=USE_EXTRA_ATTACK_FEATURES)
#     test_features = extract_features_by_class(target_model, target_test_loader, label_indicator=0, use_extra=USE_EXTRA_ATTACK_FEATURES)
#
#     for cls in range(NUM_CLASSES):
#         if cls not in attack_models:
#             continue
#
#         cls_train = train_features.get(cls, [])
#         cls_test = test_features.get(cls, [])
#
#         if not cls_train or not cls_test:
#             continue
#
#         X_train, _ = zip(*cls_train)
#         X_test, _ = zip(*cls_test)
#
#         min_len = min(len(X_train), len(X_test))
#         X_train = X_train[:min_len]
#         X_test = X_test[:min_len]
#
#         X_eval = np.vstack((X_train, X_test))
#         y_true = np.array([1] * min_len + [0] * min_len)
#
#         X_eval_scaled = scalers[cls].transform(X_eval)
#         y_pred = attack_models[cls].predict(X_eval_scaled)
#         y_scores = attack_models[cls].predict_proba(X_eval_scaled)[:, 1]
#
#         tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#         far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
#
#         print(f"Class {cls}: Accuracy={accuracy_score(y_true, y_pred):.2f}, Precision={precision_score(y_true, y_pred):.2f}, Recall={recall_score(y_true, y_pred):.2f}, F1={f1_score(y_true, y_pred):.2f}, AUC={roc_auc_score(y_true, y_scores):.2f}, FAR={far:.2f}")
#
#
# def load_highest_round_number_target_model():
#     release_model(None, DEVICE.type)
#     model = MODEL_ARCH()
#
#     # Get all checkpoint paths
#     checkpoint_paths = glob.glob(f"{TARGET_CHECKPOINT_DIR}/global_model_round_*.pth")
#
#     if not checkpoint_paths:
#         raise FileNotFoundError("No checkpoint files found.")
#
#     # Extract round numbers and map to paths
#     checkpoints_with_rounds = []
#     for path in checkpoint_paths:
#         match = re.search(r'global_model_round_(\d+)\.pth', os.path.basename(path))
#         if match:
#             round_number = int(match.group(1))
#             checkpoints_with_rounds.append((round_number, path))
#
#     if not checkpoints_with_rounds:
#         raise ValueError("No valid checkpoint files with round numbers found.")
#
#     # Get the path with the highest round number
#     latest_round, latest_checkpoint_path = max(checkpoints_with_rounds, key=lambda x: x[0])
#
#     model.load_state_dict(torch.load(latest_checkpoint_path, map_location=DEVICE))
#     model.to(DEVICE)
#     model.eval()
#     print(f"Loaded target model from: {latest_checkpoint_path} (round {latest_round})")
#
#     return model
#
#
# # === Main ===
# if __name__ == "__main__":
#     shadow_subsets, target_train, target_test, shadow_test = load_data()
#
#     shadow_data = train_shadow_models(shadow_subsets, shadow_test)
#     attack_models, scalers = train_attack_models(shadow_data)
#
#     target_model = load_highest_round_number_target_model()
#     target_train_loader = DataLoader(target_train, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker, generator=g)
#     target_test_loader = DataLoader(target_test, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker, generator=g)
#
#     evaluate_per_class_attack_models(attack_models, scalers, target_model, target_train_loader, target_test_loader)
#     release_model(target_model, DEVICE.type)
