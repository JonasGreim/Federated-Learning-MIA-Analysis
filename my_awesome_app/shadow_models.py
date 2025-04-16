from datasets import load_dataset
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from my_awesome_app.models.black_box_shadow_model import BlackBoxShadowModel
from my_awesome_app.models.simple_model import NetSimple
from sklearn.linear_model import LogisticRegression
import numpy as np
import glob
import os

# Load CIFAR-10 dataset
dataset = load_dataset('cifar10', batch_size=32)


# Apply transform
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

# split data
train_dataset = dataset["train"]
test_dataset = dataset["test"]

train_split_size = len(train_dataset) // 2
shadow_train_dataset, target_train_dataset = random_split(train_dataset, [train_split_size, train_split_size])

# load data into dataloader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)
target_train_loader = DataLoader(target_train_dataset, batch_size=64, shuffle=True)
shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=64, shuffle=True)


# train function
def train_model(model, dataloader, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            inputs, labels = batch["img"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model


shadow_model = NetSimple()
shadow_model = train_model(shadow_model, shadow_train_loader)


# Collect Confidence Scores (for attack model)
# We'll use softmax outputs as input features and whether the sample was in the training set as the label.
def get_confidences(model, dataloader, label):
    model.eval()
    device = next(model.parameters()).device
    confidences = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            inputs, true_labels = batch["img"].to(device), batch["label"].to(device)
            outputs = model(inputs)  # input = batches of data
            probs = F.softmax(outputs, dim=1)  # convert tensors to probability tensors batches
            max_conf = probs.max(dim=1)[0]  # maximum value per row (data object)
            confidences.extend(max_conf.cpu().numpy())  # moves tensor to cpu, covert to numpy array, add max. confidences to list
            labels.extend([label] * len(max_conf))  # label = 1 if member (train set), 0 if non-member (test set) -> depends on input data
    return confidences, labels


# prepare attack dataset
# For the shadow model, we know exactly which samples were members
shadow_member_conf, shadow_member_labels = get_confidences(shadow_model, shadow_train_loader, 1)
shadow_nonmember_conf, shadow_nonmember_labels = get_confidences(shadow_model, test_loader, 0)

X_attack = shadow_member_conf + shadow_nonmember_conf
y_attack = shadow_member_labels + shadow_nonmember_labels

# Train the Attack Model
X_attack = np.array(X_attack).reshape(-1, 1)
y_attack = np.array(y_attack)

attack_model = LogisticRegression().fit(X_attack, y_attack)

# Load target model
list_of_files = [fname for fname in glob.glob("../model_checkpoints/global_model_round_*")]
latest_round_file = max(list_of_files, key=os.path.getctime)

target_model = NetSimple()
target_model.load_state_dict(torch.load(f=latest_round_file, map_location='cpu'))
print("Loading pre-trained model from: ", latest_round_file)
target_model.eval()


# Run the Attack on Target Mode
# Measured how well it could infer membership

target_member_conf, _ = get_confidences(target_model, target_train_loader, 1)
target_nonmember_conf, _ = get_confidences(target_model, test_loader, 0)

X_target_attack = np.array(target_member_conf + target_nonmember_conf).reshape(-1, 1)
y_target_true = np.array([1] * len(target_member_conf) + [0] * len(target_nonmember_conf))

y_pred = attack_model.predict(X_target_attack)
attack_accuracy = np.mean(y_pred == y_target_true)
print(f"Attack Accuracy: {attack_accuracy:.2f}")
