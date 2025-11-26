import wandb
import matplotlib.pyplot as plt
from pathlib import Path

from experiments_data_analysis.data_interface.column_names_merged_table import Col
from experiments_data_analysis.data_interface.wandb_flower_table import ColFlower


def download_and_plot_flower_metrics(entity: str, project: str, output_dir: Path):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(runs)} runs")

    for run in runs:
        print(f"\n--- Run: {run.id} ({run.name}) ---")

        # output folder for this run
        run_folder = output_dir / run.name
        run_folder.mkdir(parents=True, exist_ok=True)

        # Load history
        df = run.history(samples=None)

        # x-axis selection
        if ColFlower.STEP.value in df.columns:
            x = df[ColFlower.STEP.value]
            x_label = ColFlower.STEP.value
        elif ColFlower.RUNTIME.value in df.columns:
            x = df[ColFlower.RUNTIME.value]
            x_label = ColFlower.RUNTIME.value
        else:
            x = df.index
            x_label = "index"

        # iterate only over enum-defined metrics
        for metric in ColFlower:
            if metric in (ColFlower.STEP, ColFlower.RUNTIME):
                continue  # skip x-axis

            metric_name = metric.value

            # skip missing metrics
            if metric_name not in df.columns:
                print(f"Skipping {metric_name}: not found in history")
                continue

            y = df[metric_name]

            # create plot
            fig, ax = plt.subplots()
            ax.plot(x, y)
            ax.set_xlabel("Serverrunde")
            a: str = Col[metric.name].value
            ax.set_ylabel(a)
            fig.tight_layout()

            # safe filename
            filename = f"{metric.name.lower()}.png"
            fig.savefig(run_folder / filename, dpi=200)
            plt.close(fig)

            print(f"Saved plot: {filename}")

    print("\nDone!")


