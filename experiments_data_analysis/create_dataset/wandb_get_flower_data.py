import wandb
import pandas as pd
from pathlib import Path


def export_flower_wandb_runs_to_csv(
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
        data.update(run.config)

        run_data.append(data)

    # Convert to DataFrame
    df = pd.DataFrame(run_data)

    # ✅ Choose which columns to keep
    columns_to_keep = [
        "run-name",
        "_runtime",
        "Overfitting-Gap (Clients aggregiert, Accuracy)",
        "Overfitting-Gap (Clients aggregiert, Loss)",
        "Overfitting-Gap (Server, Accuracy)",
        "Overfitting-Gap (Server, Loss)",  # example config
        "Test-Accuracy (Server)",  # example config
        "Test-Loss (Server)",  # example config
        "Trainings-Accuracy (Clients aggregiert)",  # example metric
        "Trainings-Loss (Clients aggregiert)",  # example metric
        "Validierungs-Accuracy (Clients aggregiert)",  # example config
        "Validierungs-Loss (Clients aggregiert)",  # example metric
        "model",
        "local-epochs",  # example metric
        "weight-decay",  # example config
        "dirichlet-alpha",  # example metric
        "num-server-rounds",  # example metric
        "iid-data-distribution",  # example config
        "num-clients",
    ]

    # Keep only those columns that exist in the DataFrame
    print(df.columns)

    df = df[[col for col in columns_to_keep if col in df.columns]]

    # Save to CSV
    df.to_csv(output_path, index=False)

    print(f"✅ Saved filtered CSV with {len(df)} runs and {len(df.columns)} columns -> {output_path}")
