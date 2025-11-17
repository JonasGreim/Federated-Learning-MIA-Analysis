import wandb
import pandas as pd
from pathlib import Path

from experiments_data_analysis.file_name_settings import FILE_NAME_MIA


def export_mia_wandb_runs_to_csv(
        entity: str,
        project: str,
        output_path: Path
):
    # Authenticate
    api = wandb.Api()

    # Fetch runs
    runs = api.runs(f"{entity}/{project}")

    run_data = []
    for run in runs:
        data = {}

        # Safely get summary as a dict
        summary_dict = getattr(run.summary, "_json_dict", {}) or {}
        data.update(summary_dict)

        # Keep full parameters dict as JSON string
        data.update(run.config.get("parameters", {}))

        run_data.append(data)

    # Convert to DataFrame
    df = pd.DataFrame(run_data)

    # ✅ Choose which columns to keep
    columns_to_keep = [
        "run_name",
        "_runtime",
        "parameters",
        "model_arch",
        "num_shadow_models",
        "shadow_epochs",  # example config
        "train_size",  # example config
        "weight_decay",  # example config
        "metrics/AUC",  # example metric
        "metrics/Accuracy",  # example metric
        "metrics/F1-Score",  # example config
        "metrics/Precision",  # example metric
        "metrics/Recall",  # example config
        "std/std_AUC",  # example metric
        "std/std_Accuracy",  # example config
        "std/std_F1-Score",  # example metric
        "std/std_Precision",  # example config
        "std/std_Recall",  # example metric
    ]

    # Keep only those columns that exist in the DataFrame
    print(df.columns)

    df = df[[col for col in columns_to_keep if col in df.columns]]

    # Save to CSV
    file_name = output_path / FILE_NAME_MIA
    df.to_csv(file_name, index=False)

    print(f"✅ Saved filtered CSV with {len(df)} runs and {len(df.columns)} columns -> {output_path}")
