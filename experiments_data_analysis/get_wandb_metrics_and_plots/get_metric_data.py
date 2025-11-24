import wandb
import os
from pathlib import Path


def get_wandb_metrics_data(
        entity: str,
        project: str,
        output_path: Path
):
    api = wandb.Api()

    # Fetch runs
    runs = api.runs(f"{entity}/{project}")

    print(f"Found {len(runs)} runs")

    os.makedirs(output_path, exist_ok=True)

    for run in runs:
        print(f"\n--- Run: {run.id} ({run.name}) ---")
        run_folder = output_path / run.name
        run_folder.mkdir(parents=True, exist_ok=True)

        for f in run.files():
            if f.name.startswith("run"):
                print(f"Downloading: {f.name}")
                f.download(root=run_folder, replace=True)

            if f.name.startswith("media/images/") and f.name.endswith((".png", ".jpg", ".jpeg")):
                print(f"Downloading: {f.name}")
                f.download(root=run_folder, replace=True)

    print("\nDone!")
