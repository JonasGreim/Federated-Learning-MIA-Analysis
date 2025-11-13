import pandas as pd
from experiments_data_analysis.file_name_settings import mia_data_path, flower_data_path, merge_data_file_path
from experiments_data_analysis.utils.merge_utils import map_regularization, map_model_name, map_distribution

# set output path
output_path = merge_data_file_path

# Load both CSVs
mia = pd.read_csv(mia_data_path)
flower = pd.read_csv(flower_data_path)

# Perform the join (inner join by default)
merged = flower.merge(
    mia,
    left_on="run-name",
    right_on="run_name",
    how="inner"
)

# Rename columns
merged = merged.rename(columns={
    "_runtime_y": "Runtime_MIA",
    "_runtime_x": "Runtime_Flower",
})

# Drop columns that are duplicates or not needed
drop_columns_dublicates = ["run_name", "model_arch", "weight_decay", "num-clients"]
drop_colums_side_experiments = ["num_shadow_models", "shadow_epochs", "train_size", "num-clients"]
drop_colums = drop_columns_dublicates + drop_colums_side_experiments
merged = merged.drop(columns=drop_colums)

# add regularization column (weight-decay>0.0 -> "True", else -> "False")
merged["regularization"] = merged["weight-decay"].apply(map_regularization)
merged = merged.drop(columns=["weight-decay"])

# Round all numeric columns to 2 decimal places
merged = merged.round(2)

# Map "dirichlet-alpha" values: 0.0 -> "iid", 0.5 -> "semi-noniid", 0.1 -> "noniid"
merged["data_distribution"] = merged["dirichlet-alpha"].apply(map_distribution)
merged = merged.drop(columns=["iid-data-distribution", "dirichlet-alpha"])

# Map model names for better readability
merged["model"] = merged["model"].apply(map_model_name)

# rename some colums no "," in column names
merged = merged.rename(columns={
    "Overfitting-Gap (Clients aggregiert, Accuracy)": "Overfitting-Gap (Clients aggregiert: Accuracy)",
    "Overfitting-Gap (Clients aggregiert, Loss)": "Overfitting-Gap (Clients aggregiert: Loss)",
    "Overfitting-Gap (Server, Accuracy)": "Overfitting-Gap (Server: Accuracy)",
    "Overfitting-Gap (Server, Loss)": "Overfitting-Gap (Server: Loss)",
})

# Save result
merged.to_csv(output_path, index=False)

print(merged.columns.tolist())
print("Merged shape:", merged.shape)
