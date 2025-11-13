import pandas as pd
from experiments_data_analysis.file_name_settings import merge_data_file_path, overview_file_path

# set output path
output_path = overview_file_path

# Load both CSVs
full_data_tables = pd.read_csv(merge_data_file_path)

keep_columns = ["model", "data_distribution", "local-epochs", "num-server-rounds", "metrics/AUC", "Overfitting-Gap (Server: Loss)", "Test-Accuracy (Server)"]
overview_table = full_data_tables[keep_columns]

overview_table.to_csv(output_path, index=False)
