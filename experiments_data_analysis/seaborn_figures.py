import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from experiments_data_analysis.file_name_settings import merge_data_file_path
from experiments_data_analysis.utils.merge_utils import map_regularization_diagram

# Ensure figures directory exists
os.makedirs("figures", exist_ok=True)

# Load data
merged = pd.read_csv(merge_data_file_path)



merged = merged.rename(columns={
    "model": "Modell",
    "regularization": "Regelarisierung",
    "data_distribution": "Datenverteilung",
    "local-epochs": "Lokale Epochen",
    "Overfitting-Gap (Server: Loss)": "Overfitting Gap Loss",
    "Overfitting-Gap (Server: Accuracy)": "Overfitting Gap Acc",
    "metrics/AUC": "MIA AUC",
})

# --- 1️⃣ Effect of Local Epochs ---
plt.figure(figsize=(6, 4))
sns.boxplot(data=merged, x="Lokale Epochen", y="MIA AUC")
plt.title("Einfluss der lokaler Epochen auf die MIA AUC")
plt.xlabel("Lokale Epochen")
plt.ylabel("MIA AUC")
plt.tight_layout()
plt.savefig("figures/effect_local_epochs.png", dpi=300)
plt.close()

# --- 2️⃣ Architecture Influence ---
plt.figure(figsize=(6, 4))
sns.boxplot(data=merged, x="Modell", y="MIA AUC")
plt.title("Architektonischer Einfluss auf die MIA AUC")
plt.xlabel("Modellarchitektur")
plt.ylabel("MIA AUC")
plt.tight_layout()
plt.savefig("figures/architecture_influence.png", dpi=300)
plt.close()

# --- 3️⃣ Data Distribution Influence ---
plt.figure(figsize=(6, 4))
sns.boxplot(data=merged, x="Datenverteilung", y="MIA AUC", order=["IID", "semi-non-IID", "non-IID"])
plt.title("Einfluss der Datenverteilung auf die MIA AUC")
plt.xlabel("Datenverteilung")
plt.ylabel("MIA AUC")
plt.tight_layout()
plt.savefig("figures/data_distribution_influence.png", dpi=300)
plt.close()

# --- 4️⃣ Interaction: Architecture × Data Distribution ---
g = sns.catplot(
    data=merged,
    x="Datenverteilung",
    y="MIA AUC",
    hue="Modell",
    kind="bar",
    height=5, aspect=1.2
)
g.figure.suptitle("MIA AUC: Architektur × Datenverteilung", y=1.02)
g.set_axis_labels("Datenverteilung", "MIA AUC")
g.savefig("figures/interaction_architecture_data.png", dpi=300)
plt.close()




# --- 5 Regularization Influence ---
merged["Regelarisierung"] = merged["Regelarisierung"].apply(map_regularization_diagram)
plt.figure(figsize=(6, 4))
sns.boxplot(data=merged, x="Regelarisierung", y="MIA AUC")
plt.title("Einfluss von Regelarisierung auf die MIA AUC")
# plt.xlabel("Regelarisierung")
plt.ylabel("MIA AUC")
plt.tight_layout()
plt.savefig("figures/regularization_influence.png", dpi=300)
plt.close()



plt.figure(figsize=(10, 4))
g = sns.catplot(
    data=merged,
    x="Regelarisierung",
    y="MIA AUC",
    col="Modell",
    kind="box",
    sharey=True,
    height=4,
    aspect=1
)

g.fig.suptitle("Einfluss der Regularisierung getrennt nach Modellarchitektur", y=1.05)

plt.savefig("figures/regularization_per_model.png", dpi=300)
plt.close()