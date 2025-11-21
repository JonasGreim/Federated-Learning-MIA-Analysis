import matplotlib.pyplot as plt
from experiments_data_analysis.data_interface.column_names_merged_table import Col
from pathlib import Path
import pandas as pd


def create_shadow_model_number_figures(data_path: Path, figures_dir: Path):
    overview_table = pd.read_csv(data_path)

    plt.figure(figsize=(8, 5))

    # Ensure numeric x-axis
    overview_table[Col.NUM_SHADOW_MODELS] = overview_table[Col.NUM_SHADOW_MODELS].astype(int)

    # Expected categories (ensure consistent spelling)
    models = ["Shokri-CNN", "ResNet-18"]
    distributions = ["IID", "non_IID"]

    for model in models:
        for dist in distributions:
            subset = overview_table[
                (overview_table[Col.MODEL] == model) &
                (overview_table[Col.DATA_DISTRIBUTION] == dist)
                ]

            if subset.empty:
                print(f"WARNING: Missing data for {model} + {dist}")
                continue

            subset = subset.sort_values(by=Col.NUM_SHADOW_MODELS)

            label = f"{model} ({dist})"
            plt.plot(
                subset[Col.NUM_SHADOW_MODELS],
                subset[Col.METRICS_AUC],
                marker="o",
                label=label,
            )

    plt.xlabel("Number of Shadow Models")
    plt.ylabel("MIA AUC")
    plt.title("Shadow Model Experiments")
    plt.xticks([1, 3, 10, 20])
    plt.legend()
    plt.tight_layout()

    plt.savefig(figures_dir / "auc_per_shadow_model_number_per_model.png", dpi=300)
    plt.close()
