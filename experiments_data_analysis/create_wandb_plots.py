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

        df.save_csv(run_folder / "wandb_history.csv")

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
            fig, ax = plt.subplots(figsize=(8, 6))

            # Line with same color and thicker width
            ax.plot(x, y, color='#1f77b4', linewidth=2.2)

            # Axis style like your bar plots
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            metrics_from_0_to_1 = {
                ColFlower.VALIDIERUNGS_ACCURACY_CLIENTS_AGGREGIERT,
                ColFlower.TRAININGS_ACCURACY_CLIENTS_AGGREGIERT,
                ColFlower.TEST_ACCURACY_SERVER,
                ColFlower.OVERFITTING_GAP_ACC,
                ColFlower.OVERFITTING_GAP_ACC_CLIENTS,
            }
            if metric in metrics_from_0_to_1:
                plt.ylim(0, 1.02)  # small margin above 1

            # labels
            ax.set_xlabel("Serverrunde", fontsize=12)
            ylabel = Col[metric.name].value
            ax.set_ylabel(ylabel, fontsize=12)

            # grid & tick styling
            ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.3)
            ax.tick_params(axis='both', labelsize=11)

            # layout
            fig.tight_layout()

            # safe filename
            filename = f"{metric.name.lower()}.png"
            fig.savefig(run_folder / filename, dpi=300)
            plt.close(fig)

            print(f"Saved plot: {filename}")

    print("\nDone!")


