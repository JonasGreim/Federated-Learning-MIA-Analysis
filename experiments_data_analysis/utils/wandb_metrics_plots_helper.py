import os
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
from experiments_data_analysis.data_interface.column_names_merged_table import Col
from experiments_data_analysis.data_interface.wandb_flower_table import ColFlower


def process_run_files(run, run_folder: Path, images_folder: Path) -> None:
    """Download metric JSONs and media images for a single W&B run."""
    for f in run.files():
        is_metric_file = f.name.startswith("run") and f.name.endswith(".json")
        is_image_file = (
                f.name.startswith("media/images/")
                and "per_class" not in f.name
                and "overall_metrics" not in f.name
                and f.name.lower().endswith((".png", ".jpg", ".jpeg"))
        )

        if not (is_metric_file or is_image_file):
            continue

        print(f"Downloading: {f.name}")

        f.download(root=run_folder, replace=True)
        downloaded_path = run_folder / f.name

        final_dir = images_folder if is_image_file else run_folder
        final_path = final_dir / os.path.basename(f.name)

        if final_path.exists():
            base, ext = os.path.splitext(final_path.name)
            i = 1
            while (final_dir / f"{base}_{i}{ext}").exists():
                i += 1
            final_path = final_dir / f"{base}_{i}{ext}"

        shutil.move(str(downloaded_path), str(final_path))

    for dirpath, dirnames, filenames in os.walk(run_folder, topdown=False):
        if Path(dirpath) == run_folder:
            continue
        try:
            os.rmdir(dirpath)
        except OSError:
            pass


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

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(x, y, color="#1f77b4", linewidth=2.2)

        # Achsen-Stil
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # y-Limit für Metriken in [0, 1]
        if metric in metrics_from_0_to_1:
            ax.set_ylim(0, 1.02)

        # Beschriftungen
        ax.set_xlabel(x_label, fontsize=12)
        ylabel = Col[metric.name].value
        ax.set_ylabel(ylabel, fontsize=12)

        # Grid + Ticks
        ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.3)
        ax.tick_params(axis="both", labelsize=11)

        fig.tight_layout()

        filename = f"{metric.name.lower()}.png"
        fig.savefig(images_folder / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved plot: images/{filename}")
