import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from experiments_data_analysis.file_name_settings import merge_data_file_path

# Load your merged data
merged = pd.read_csv(merge_data_file_path)

merged = merged.rename(columns={
    "model": "Model",
    "regularization": "Reg.",
    "data_distribution": "Data Dist.",
    "local-epochs": "Local Epochs",
    "metrics/AUC": "MIA AUC",
    "Overfitting-Gap (Server: Loss)": "Overfitting Gap Loss",
    "Overfitting-Gap (Server: Accuracy)": "Overfitting Gap Acc",
    "Overfitting-Gap (Clients aggregiert: Accuracy)": "Overfitting Gap Loss (Clients)",
    "Overfitting-Gap (Clients aggregiert: Loss)": "Overfitting Gap Acc (Clients)",
})


# Select columns of interest
cols_of_interest = [
    "MIA AUC",
    "Overfitting Gap Loss",
    "Overfitting Gap Acc",
    "Overfitting Gap Loss (Clients)",
    "Overfitting Gap Acc (Clients)"
]

# Compute correlation matrix
corr = merged[cols_of_interest].corr()

# Print numeric correlation table
print(corr.round(2))

# Plot heatmap
plt.figure(figsize=(7,5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Korrelation zwischen den Overfitting Gaps und der MIA AUC")
plt.tight_layout()
plt.savefig("figures/correlation_overfitting_auc.png", dpi=300)
plt.close()



g = sns.pairplot(
    merged,
    vars=[
        "MIA AUC",
        "Overfitting Gap Loss"
    ],
    diag_kind="kde"
)

# Add title and adjust space above plots
g.figure.suptitle("Korrelation von Overfitting Gap Loss und MIA-AUC", y=1.03)
g.figure.subplots_adjust(top=0.95)  # increase or decrease top margin

# Save
g.savefig("figures/pairplot_overfitting_auc.png", dpi=300)
plt.close()


g = sns.pairplot(
    merged,
    vars=[
        "MIA AUC",
        "Overfitting Gap Acc (Clients)"
    ],
    diag_kind="kde"
)

# Add title and adjust space above plots
g.figure.suptitle("Korrelation von Overfitting Gap Acc und MIA-AUC", y=1.03)
g.figure.subplots_adjust(top=0.95)  # increase or decrease top margin

# Save
g.savefig("figures/pairplot_overfitting_auc_acc.png", dpi=300)
plt.close()
