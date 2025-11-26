import wandb
import os
from pathlib import Path
from experiments_data_analysis.utils.wandb_metrics_plots_helper import process_run_files, generate_flower_plots_for_run


def get_wandb_metrics_and_plots(
        entity: str,
        project: str,
        output_path: Path,
        generate_wandb_plots: bool = False,
):
    api = wandb.Api()

    runs = api.runs(f"{entity}/{project}")
    print(f"Found {len(runs)} runs")

    os.makedirs(output_path, exist_ok=True)

    for run in runs:
        print(f"\n--- Run: {run.id} ({run.name}) ---")

        run_name = run.name.split("/", 1)[0] if "/" in run.name else run.name
        run_folder = output_path / run_name
        run_folder.mkdir(parents=True, exist_ok=True)

        images_folder = run_folder / "images"
        images_folder.mkdir(exist_ok=True)

        # -----------------------------
        # 1) Download files from wandb
        # -----------------------------
        process_run_files(run, run_folder, images_folder)

        # ----------------------------------
        # 2) Load wandb history and generate plots
        # ----------------------------------
        if generate_wandb_plots:
            generate_flower_plots_for_run(run, run_folder, images_folder)



    print("\nDone!")
