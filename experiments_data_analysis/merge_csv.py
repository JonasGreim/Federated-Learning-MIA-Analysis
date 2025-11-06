import pandas as pd
import math

# Load both CSVs
mia = pd.read_csv("./wandb_mia.csv")
flower = pd.read_csv("./wandb_flower.csv")

# Perform the join (inner join by default)
merged = flower.merge(
    mia,
    left_on="run_name",
    right_on="parameters.run_name",
    how="inner"
)

# Rename columns (example)
merged = merged.rename(columns={
    "Runtime_y": "Runtime_MIA",
    "Runtime_x": "Runtime_Flower",
})

# Optional: drop redundant join key if needed
drop_columns = ["Name_x", "Name_y", "parameters.model_arch", "parameters.num_shadow_models", "parameters.shadow_epochs",
               "parameters.target_model_file", "parameters.test_size", "parameters.train_size", "num-clients", "parameters.run_name"]
merged = merged.drop(columns=drop_columns)


def map_regularization(weight_decay):
    return math.isclose(weight_decay, 0.0005, rel_tol=1e-9)


merged["regularization"] = merged["parameters.weight_decay"].apply(map_regularization)

merged = merged.round(2)

# "dirichlet-alpha" => 0.0 = iid, 0.5 = semi-noniid, 0.1 = noniid
def map_distribution(alpha):
    if alpha == 0.0:
        return "IID"
    elif alpha == 0.5:
        return "semi-non-IID"
    elif alpha == 0.1:
        return "non-IID"
    else:
        return f"unknown ({alpha})"

merged["data_distribution"] = merged["dirichlet-alpha"].apply(map_distribution)
merged = merged.drop(columns=["iid-data-distribution", "dirichlet-alpha"])
merged = merged.drop(columns=["Runtime_MIA", "Runtime_Flower", "parameters.weight_decay", "weight-decay"])


def map_model_name(model):
    if model == "simple_model":
        return "Shokri-CNN"
    elif model == "simple_model_with_dropout":
        return "Shokri-CNN"
    elif model == "complex_model":
        return "ResNet-18"
    elif model == "complex_model_with_dropout":
        return "ResNet-18"
    else:
        return f"unknown ({model})"

merged["model"] = merged["model"].apply(map_model_name)

# Save result
merged.to_csv("merged_output.csv", index=False)
print("Merged shape:", merged.shape)


summary_cols = [
    "model", "local-epochs", "dirichlet-alpha", "metrics/AUC", "Test-Accuracy (Server)"
]
