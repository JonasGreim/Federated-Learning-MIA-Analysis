import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure figures directory exists
os.makedirs("figures", exist_ok=True)

# Load data
merged = pd.read_csv("merged_output.csv")

# --- 1️⃣ Effect of Local Epochs ---
plt.figure(figsize=(6, 4))
sns.boxplot(data=merged, x="local-epochs", y="metrics/AUC")
plt.title("Effect of Local Epochs on MIA Success")
plt.xlabel("Local Epochs")
plt.ylabel("MIA AUC")
plt.tight_layout()
plt.savefig("figures/effect_local_epochs.png", dpi=300)
plt.close()

# --- 2️⃣ Architecture Influence ---
plt.figure(figsize=(6, 4))
sns.boxplot(data=merged, x="model", y="metrics/AUC")
plt.title("Architecture Influence on MIA AUC")
plt.xlabel("Model Architecture")
plt.ylabel("MIA AUC")
plt.tight_layout()
plt.savefig("figures/architecture_influence.png", dpi=300)
plt.close()

# --- 3️⃣ Data Distribution Influence ---
plt.figure(figsize=(6, 4))
sns.boxplot(data=merged, x="data_distribution", y="metrics/AUC")
plt.title("Data Distribution Influence on MIA AUC")
plt.xlabel("Data Distribution")
plt.ylabel("MIA AUC")
plt.tight_layout()
plt.savefig("figures/data_distribution_influence.png", dpi=300)
plt.close()

# --- 4️⃣ Interaction: Architecture × Data Distribution ---
g = sns.catplot(
    data=merged,
    x="data_distribution",
    y="metrics/AUC",
    hue="model",
    kind="bar",
    height=5, aspect=1.2
)
g.fig.suptitle("Interaction: Architecture × Data Distribution", y=1.02)
g.set_axis_labels("Data Distribution", "Mean MIA AUC")
g.savefig("figures/interaction_architecture_data.png", dpi=300)
plt.close()
