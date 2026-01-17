import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


def create_shadow_model_number_figures(data_path: Path, figures_dir: Path):
    overview_table = pd.read_csv(data_path)

    plt.figure()

    # Ensure numeric x-axis
    overview_table["Nsh"] = overview_table["Nsh"].astype(int)

    # Expected categories (ensure consistent spelling)
    models = ["Shokri-CNN", "ResNet-18"]
    distributions = ["IID", "non-IID"]

    for model in models:
        for dist in distributions:
            subset = overview_table[
                (overview_table["Mod"] == model) &
                (overview_table["Dist"] == dist)
                ]
            print("Dist unique:", overview_table["Dist"].unique())
            print(overview_table["Dist"].value_counts(dropna=False))
            if subset.empty:
                print(f"WARNING: Missing data for {model} + {dist}")
                continue

            subset = subset.sort_values(by="Nsh")

            label = f"{model} ({dist})"
            plt.plot(
                subset["Nsh"],
                subset["AUC"],
                marker="o",
                label=label,
            )
    ax = plt.gca()
    ax.grid(axis="y")
    plt.xlabel("Anzahl der Shadow Models")
    plt.ylabel("MIA-AUC")
    plt.xticks([1, 3, 10, 20])

    legend_labels = {
        "Shokri-CNN (IID)": "Shokri-CNN (IID)",
        "Shokri-CNN (non_IID)": "Shokri-CNN (non-IID)",
        "ResNet-18 (IID)": "ResNet-18 (IID)",
        "ResNet-18 (non_IID)": "ResNet-18 (non-IID)",
    }
    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = [legend_labels.get(lbl, lbl) for lbl in labels]
    plt.legend(
        handles,
        new_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        title="Modell (Datenverteilung)",
    )
    plt.tight_layout()
    plt.savefig(figures_dir / "auc_per_shadow_model_number_per_model.png", dpi=300)
    plt.close()

    print("✅ Overview table (shadow model number) created successfully.")
