from pathlib import Path
import matplotlib.pyplot as plt
from experiments_data_analysis.data_interface.column_names_merged_table import Col
from experiments_data_analysis.data_interface.wandb_flower_table import ColFlower


def generate_flower_plots_for_run(run, run_folder: Path, images_folder: Path) -> None:
    """Load W&B history for a run and generate Flower metric plots into images/."""
    df = run.history(samples=None)

    if df.empty:
        print("No history found for this run, skipping plots.")
        return

    (run_folder / "wandb_history.csv").write_text(df.to_csv(index=False))

    if ColFlower.STEP.value in df.columns:
        x = df[ColFlower.STEP.value]
    elif ColFlower.RUNTIME.value in df.columns:
        x = df[ColFlower.RUNTIME.value]
    else:
        x = df.index
    x_label = "Serverrunde"

    metrics_from_0_to_1 = {
        ColFlower.VALIDIERUNGS_ACCURACY_CLIENTS_AGGREGIERT,
        ColFlower.TRAININGS_ACCURACY_CLIENTS_AGGREGIERT,
        ColFlower.TEST_ACCURACY_SERVER,
        ColFlower.OVERFITTING_GAP_ACC,
        ColFlower.OVERFITTING_GAP_ACC_CLIENTS,
    }

    for metric in ColFlower:
        if metric in (ColFlower.STEP, ColFlower.RUNTIME):
            continue

        metric_name = metric.value

        if metric_name not in df.columns:
            print(f"Skipping {metric_name}: not found in history")
            continue

        y = df[metric_name]

        fig, ax = plt.subplots()
        ax.plot(x, y)

        # y-Limit für Metriken in [0, 1]
        if metric in metrics_from_0_to_1:
            ax.set_ylim(0, 1.02)

        # Beschriftungen
        ax.set_xlabel(x_label)
        ylabel = Col[metric.name].value
        ax.set_ylabel(ylabel)

        # Grid + Ticks
        ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.3)
        ax.tick_params(axis="both", labelsize=11)

        fig.tight_layout()

        filename = f"{metric.name.lower()}.png"
        fig.savefig(images_folder / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved plot: images/{filename}")
