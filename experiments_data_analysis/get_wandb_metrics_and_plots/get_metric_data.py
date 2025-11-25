import wandb
import os
from pathlib import Path
import shutil


def get_wandb_metrics_data(
        entity: str,
        project: str,
        output_path: Path
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

        for f in run.files():
            is_metric_file = f.name.startswith("run") and f.name.endswith(".json")
            is_image_file = (
                    f.name.startswith("media/images/")
                    and f.name.lower().endswith((".png", ".jpg", ".jpeg"))
            )

            if not (is_metric_file or is_image_file):
                continue

            print(f"Downloading: {f.name}")

            f.download(root=run_folder, replace=True)
            downloaded_path = run_folder / f.name

            if is_image_file:
                final_dir = images_folder
            else:
                final_dir = run_folder

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

    print("\nDone!")
