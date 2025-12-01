import wandb
import os
from pathlib import Path
from experiments_data_analysis.get_wandb_metrics_and_plots.create_wandb_flower_plots import \
    generate_flower_plots_for_run
from experiments_data_analysis.get_wandb_metrics_and_plots.create_wandb_mia_plots import log_per_class_metrics2, \
    log_overall_metrics_with_error_bars2
from experiments_data_analysis.get_wandb_metrics_and_plots.process_wandb_metric_files import process_run_files


def get_wandb_metrics_and_plots(
        entity: str,
        project: str,
        output_path: Path,
        flower_run: bool = False,
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
        if flower_run:
            generate_flower_plots_for_run(run, run_folder, images_folder)
        else:
            class_names: list[str] = run.config["parameters_static"]["class_names"]
            log_per_class_metrics2(run_folder=run_folder, images_folder=images_folder,
                                   class_names=class_names)
            log_overall_metrics_with_error_bars2(run_folder=run_folder, images_folder=images_folder)

    print("\nDone!")
