import json
from experiments_data_analysis.config_plot_style import use_thesis_style
from experiments_data_analysis.file_name_settings import FIGURES_DIR_MAIN, METRICS_ANALYSIS
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def append_mia_auc_jsonl(run_name: str, mia_path: Path, out_path: Path) -> None:
    """Append (run, class, auc) rows to a JSONL file."""
    with mia_path.open("r") as f:
        data = json.load(f)

    with out_path.open("a") as out:
        for cls_str, metrics in data.items():
            out.write(json.dumps({
                "run": run_name,
                "class": int(cls_str),
                "auc": metrics.get("AUC", None),
            }) + "\n")


def log_aggregated_mia_auc_per_class(
    jsonl_path: Path,
    images_folder: Path,
    class_names: list[str] | None = None,
) -> None:
    use_thesis_style()
    """Bar plot of mean AUC per class aggregated over all runs (no std/error bars)."""
    images_folder.mkdir(parents=True, exist_ok=True)

    df = pd.read_json(jsonl_path, lines=True)
    df["class"] = df["class"].astype(int)
    df["auc"] = pd.to_numeric(df["auc"], errors="coerce")
    df = df.dropna(subset=["auc"])

    summary = (
        df.groupby("class")["auc"]
          .mean()
          .sort_index()
    )

    classes = summary.index.tolist()
    if class_names is None:
        class_names = [str(c) for c in classes]

    means = summary.values

    plt.figure()
    plt.bar(
        class_names,
        means,
        edgecolor="black",
        linewidth=1.0,
    )

    plt.xlabel("Klasse")
    plt.ylabel("MIA-AUC")
    plt.xticks(rotation=45)
    plt.ylim(0, 1.0)
    plt.grid(axis="y")

    ax = plt.gca()
    ax.set_axisbelow(False)

    plt.tight_layout()
    plt.savefig(images_folder / "per_class_auc_mean.png", dpi=300, bbox_inches="tight")
    plt.close()




log_aggregated_mia_auc_per_class(
    jsonl_path= METRICS_ANALYSIS / "main_experiments"/ "mia" / "mia_auc_all_runs.jsonl",
    images_folder=FIGURES_DIR_MAIN,
    class_names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
)



def log_aggregated_mia_auc_per_class_even_odd(
    jsonl_path: Path,
    images_folder: Path,
    class_names: list[str] | None = None,
) -> None:
    """
    Bar plots of mean MIA-AUC per class, aggregated separately over
    even and odd runs.
    """
    use_thesis_style()
    images_folder.mkdir(parents=True, exist_ok=True)

    df = pd.read_json(jsonl_path, lines=True)

    # --- extract run index (e.g. run3-model_ckp:3 -> 3)
    df["run_idx"] = (
        df["run"]
        .astype(str)
        .str.extract(r"\brun(\d+)\b", expand=False)
        .astype(int)
    )

    df["class"] = df["class"].astype(int)
    df["auc"] = pd.to_numeric(df["auc"], errors="coerce")
    df = df.dropna(subset=["auc"])

    # --- split even / odd
    splits = {
        "even": df[df["run_idx"] % 2 == 0],
        "odd":  df[df["run_idx"] % 2 == 1],
    }

    for label, df_split in splits.items():
        summary = (
            df_split.groupby("class")["auc"]
            .mean()
            .sort_index()
        )

        classes = summary.index.tolist()
        if class_names is None:
            plot_class_names = [str(c) for c in classes]
        else:
            plot_class_names = class_names

        means = summary.values

        plt.figure()
        plt.bar(
            plot_class_names,
            means,
            edgecolor="black",
            linewidth=1.0,
        )

        plt.xlabel("Klasse")
        plt.ylabel("MIA-AUC")
        plt.xticks(rotation=45)
        plt.ylim(0, 1.0)
        plt.grid(axis="y")

        ax = plt.gca()
        ax.set_axisbelow(False)

        plt.tight_layout()
        plt.savefig(
            images_folder / f"per_class_auc_mean_{label}_runs.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

log_aggregated_mia_auc_per_class_even_odd(
    jsonl_path= METRICS_ANALYSIS / "main_experiments"/ "mia" / "mia_auc_all_runs.jsonl",
    images_folder=FIGURES_DIR_MAIN,
    class_names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
)