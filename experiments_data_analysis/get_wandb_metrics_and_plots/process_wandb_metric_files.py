import os
import shutil
from pathlib import Path


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
