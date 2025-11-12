import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from experiments_data_analysis.file_name_settings import merge_file_name

# Load your merged data
merged = pd.read_csv(merge_file_name)

# Select columns of interest
cols_of_interest = [
    "metrics/AUC",
    "Overfitting-Gap (Server: Accuracy)",
    "Overfitting-Gap (Server: Loss)",
    "Overfitting-Gap (Clients aggregiert: Accuracy)",
    "Overfitting-Gap (Clients aggregiert: Loss)"
]

# Compute correlation matrix
corr = merged[cols_of_interest].corr()

# Print numeric correlation table
print(corr.round(2))

# Plot heatmap
plt.figure(figsize=(7,5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Overfitting and MIA AUC")
plt.tight_layout()
plt.savefig("figures/correlation_overfitting_auc.png", dpi=300)
plt.close()



g = sns.pairplot(
    merged,
    vars=[
        "metrics/AUC",
        "Overfitting-Gap (Server: Loss)"
    ],
    diag_kind="kde"
)

# Add title and adjust space above plots
g.figure.suptitle("Pairwise Relationships Between Overfitting and MIA AUC", y=1.03)
g.figure.subplots_adjust(top=0.95)  # increase or decrease top margin

# Save
g.savefig("figures/pairplot_overfitting_auc.png", dpi=300)
plt.close()
