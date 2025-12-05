from pathlib import Path
from typing import List
import pandas as pd
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
        ax.grid(axis="y")
        ax.tick_params(axis="both", labelsize=11)

        fig.tight_layout()

        filename = f"{metric.name.lower()}.png"
        fig.savefig(images_folder / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved plot: images/{filename}")


def generate_flower_compare_runs_same_metric_from_csv(
    run_csv_paths: List[Path],
    run_labels: List[str],
    images_folder: Path,
    metric_column_name: ColFlower,
) -> None:
    """
    Plot the same metric from multiple runs (CSV history files) in a single figure.
    Assumes all runs share the same x-index or STEP/RUNTIME.
    """

    if len(run_csv_paths) != len(run_labels):
        raise ValueError("run_csv_paths and run_labels must have same length")

    # Read CSVs into DataFrames
    dfs = [pd.read_csv(p) for p in run_csv_paths]

    if dfs[0].empty:
        print("First run CSV is empty, cannot plot.")
        return

    # --- Shared x-axis from first run ---
    df0 = dfs[0]

    if ColFlower.STEP.value in df0.columns:
        x = df0[ColFlower.STEP.value]
    elif ColFlower.RUNTIME.value in df0.columns:
        x = df0[ColFlower.RUNTIME.value]
    else:
        x = df0.index

    x_label = "Serverrunde"

    # Which column to plot
    metric_col = metric_column_name.value

    # Validate column exists
    for df, label in zip(dfs, run_labels):
        if metric_col not in df.columns:
            print(f"Skipping run '{label}': metric '{metric_col}' not found.")
            return

    # y-data values
    ys = [df[metric_col] for df in dfs]

    # Plot
    fig, ax = plt.subplots()

    for label, y in zip(run_labels, ys):
        ax.plot(x, y, label=label)

    # y-limit for metrics in range [0,1]
    metrics_from_0_to_1 = {
        ColFlower.VALIDIERUNGS_ACCURACY_CLIENTS_AGGREGIERT,
        ColFlower.TRAININGS_ACCURACY_CLIENTS_AGGREGIERT,
        ColFlower.TEST_ACCURACY_SERVER,
        ColFlower.OVERFITTING_GAP_ACC,
        ColFlower.OVERFITTING_GAP_ACC_CLIENTS,
    }

    if metric_column_name in metrics_from_0_to_1:
        ax.set_ylim(0, 1.02)

    # Labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(Col[metric_column_name.name].value)

    # Legend + Grid
    ax.legend()
    ax.grid(axis="y")
    ax.tick_params(axis="both", labelsize=11)

    fig.tight_layout()

    filename = f"compare_runs_{metric_column_name.name.lower()}.png"
    fig.savefig(images_folder / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved multi-run comparison plot: images/{filename}")




