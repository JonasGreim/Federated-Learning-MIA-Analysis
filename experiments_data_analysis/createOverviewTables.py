import pandas as pd
from experiments_data_analysis.file_name_settings import merge_data_file_path, overview_file_path, table_file_latex_path

# set output path
output_path = overview_file_path
output_path_latex = table_file_latex_path

# Load both CSVs
full_data_tables = pd.read_csv(merge_data_file_path)

keep_columns = [
    "model", "regularization", "data_distribution", "local-epochs",
    "Overfitting-Gap (Server: Loss)", "Overfitting-Gap (Server: Accuracy)", "metrics/AUC"
]

overview_table = full_data_tables[keep_columns]


overview_table = overview_table.rename(columns={
    "model": "Model",
    "regularization": "Reg.",
    "data_distribution": "Data Dist.",
    "local-epochs": "Local Epochs",
    "Overfitting-Gap (Server: Loss)": "Overfitting Gap Loss",
    "Overfitting-Gap (Server: Accuracy)": "Overfitting Gap Acc",
    "metrics/AUC": "MIA AUC",
})

# Ensure categorical sorting
overview_table["Model"] = pd.Categorical(
    overview_table["Model"], categories=["Shokri-CNN", "ResNet-18"], ordered=True
)
overview_table["Reg."] = pd.Categorical(
    overview_table["Reg."], categories=[False, True], ordered=True
)
overview_table["Data Dist."] = pd.Categorical(
    overview_table["Data Dist."],
    categories=["IID", "semi-non-IID", "non-IID"],
    ordered=True
)
overview_table["Local Epochs"] = pd.Categorical(
    overview_table["Local Epochs"], categories=[1, 5, 10], ordered=True
)

# Sort in a meaningful grid order
overview_table = overview_table.sort_values(
    ["Model", "Reg.", "Data Dist.", "Local Epochs"]
)


overview_table.to_csv(output_path, index=False)


# to latex format
latex_table = overview_table.to_latex(index=False, float_format="%.2f")

with open(output_path_latex, "w") as f:
    f.write(latex_table)
